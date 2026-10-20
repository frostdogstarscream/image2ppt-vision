import hmac
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from .contracts import Request
from .storage import Store, MAX_BYTES, read_json, digest
from .worker import Worker
from .plans import validate_plan
from . import __version__


def create_app(root=None, token=None, run_worker=True):
    store=Store(root or os.environ.get("VISION_DATA","vision-data"))
    secret=token or os.environ.get("VISION_TOKEN")
    if not secret:raise RuntimeError("VISION_TOKEN required")
    worker=Worker(store)
    @asynccontextmanager
    async def lifespan(app):
        if run_worker:worker.start()
        yield
        worker.stop()
    app=FastAPI(title="image2ppt-vision",version=__version__,lifespan=lifespan)
    def auth(authorization: str=Header(default="")):
        if not hmac.compare_digest(authorization,"Bearer "+secret):raise HTTPException(401,"unauthorized")
    @app.get("/health")
    def health():return {"status":"alive"}
    @app.get("/v1/capabilities",dependencies=[Depends(auth)])
    def capabilities():
        from .models import capabilities
        return capabilities()
    @app.post("/v1/assets",dependencies=[Depends(auth)])
    async def upload(file:UploadFile=File(...)):
        path=None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                path=Path(f.name);size=0
                while block:=await file.read(1024*1024):
                    size+=len(block)
                    if size>MAX_BYTES:raise HTTPException(413,"image too large")
                    f.write(block)
            return store.upload(path)
        except (ValueError,OSError,Image.DecompressionBombError):raise HTTPException(422,"invalid image or insufficient storage")
        finally:
            if path:path.unlink(missing_ok=True)
            await file.close()
    @app.post("/v1/jobs",status_code=202,dependencies=[Depends(auth)])
    def submit(r:Request):
        try:
            if r.operation == "reconstruct":
                asset = read_json(store.path("assets",r.asset_id)/"asset.json")
                validate_plan(r.plan,source_sha256=asset["original_sha256"],image_size=asset["size"])
            if r.operation == "reconstruct_elements":
                from .element_service import prepare
                prepare(store,r)
            return store.submit(r)
        except FileNotFoundError:raise HTTPException(404,"asset not found")
        except ValueError as e:raise HTTPException(429 if str(e)=="queue full" else 422,str(e))
    @app.get("/v1/jobs/{id}",dependencies=[Depends(auth)])
    def status(id:str):
        try:
            result=store.status(id);manifest=store.path('jobs',id)/'manifest.json'
            if manifest.is_file() and result['status'] not in {'QUEUED','RUNNING'}:result['artifacts']=read_json(manifest)
            return result
        except (ValueError,FileNotFoundError):raise HTTPException(404,"job not found")
    @app.get("/v1/jobs/{id}/artifacts/{name:path}",dependencies=[Depends(auth)])
    def artifact(id:str,name:str):
        try:
            directory=store.path("jobs",id);manifest=read_json(directory/"manifest.json")
            if name not in manifest:raise FileNotFoundError()
            path=(directory/name).resolve()
            if not path.is_relative_to(directory) or digest(path)!=manifest[name]["sha256"]:raise ValueError()
            return FileResponse(path)
        except (ValueError,FileNotFoundError):raise HTTPException(404,"verified artifact unavailable")
    return app
