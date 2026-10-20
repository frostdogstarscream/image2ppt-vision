import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageOps

MAX_BYTES = 40_000_000
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for d in ("assets", "jobs"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,status TEXT NOT NULL,created TEXT NOT NULL)")

    def db(self):
        db = sqlite3.connect(self.root / "index.sqlite3", timeout=5)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def space(self):
        if shutil.disk_usage(self.root).free < 20 * 1024**3:
            raise ValueError("disk reserve below 20 GiB")

    def path(self, kind, id):
        if not re.fullmatch(r"[a-f0-9]{32}", id):
            raise ValueError("invalid resource ID")
        p = self.root / kind / id
        if not p.is_dir():
            raise FileNotFoundError("resource not found")
        return p

    def upload(self, source):
        self.space()
        source = Path(source)
        if source.stat().st_size > MAX_BYTES:
            raise ValueError("image exceeds 40 MB")
        with Image.open(source) as im:
            if im.format not in {"PNG", "JPEG", "WEBP"} or getattr(im, "n_frames", 1) != 1:
                raise ValueError("only single-frame PNG/JPEG/WebP supported")
            if im.width * im.height > MAX_PIXELS:
                raise ValueError("image exceeds 40 million pixels")
            size = list(im.size)
            orientation = im.getexif().get(274, 1)
            normalized = ImageOps.exif_transpose(im).convert("RGBA")
            id = uuid.uuid4().hex
            target = self.root / "assets" / id
            target.mkdir()
            shutil.copyfile(source, target / "original")
            normalized.save(target / "image.png")
        record = {"id": id, "original_sha256": digest(source), "sha256": digest(target / "image.png"),
                  "original_size": size, "size": list(normalized.size), "exif_orientation": orientation,
                  "coordinate_space": "exif_transposed_source_pixels"}
        w,h=size
        matrices={1:[[1,0,0],[0,1,0],[0,0,1]],2:[[-1,0,w],[0,1,0],[0,0,1]],
                  3:[[-1,0,w],[0,-1,h],[0,0,1]],4:[[1,0,0],[0,-1,h],[0,0,1]],
                  5:[[0,1,0],[1,0,0],[0,0,1]],6:[[0,-1,h],[1,0,0],[0,0,1]],
                  7:[[0,-1,h],[-1,0,w],[0,0,1]],8:[[0,1,0],[-1,0,w],[0,0,1]]}
        record["original_to_normalized_edges_matrix"]=matrices.get(orientation,matrices[1])
        write_json(target / "asset.json", record)
        return record

    def submit(self, request):
        self.space()
        for id in (request.asset_id, request.reference_id, request.mask_id):
            if id:
                self.path("assets", id)
        id = uuid.uuid4().hex
        created = datetime.now(timezone.utc).isoformat()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM jobs WHERE status='QUEUED'").fetchone()[0] >= 20:
                raise ValueError("queue full")
            target = self.root / "jobs" / id
            target.mkdir()
            write_json(target / "request.json", request.model_dump())
            write_json(target / "status.json", {"id": id, "status": "QUEUED", "created": created})
            db.execute("INSERT INTO jobs VALUES(?,?,?)", (id, "QUEUED", created))
        return self.status(id)

    def status(self, id):
        return read_json(self.path("jobs", id) / "status.json")

    def update(self, id, status, **data):
        record = self.status(id)
        record.update(status=status, **data)
        write_json(self.path("jobs", id) / "status.json", record)
        with self.db() as db:
            db.execute("UPDATE jobs SET status=? WHERE id=?", (status, id))

    def recover(self):
        # Files are authoritative; rebuild derived index entries, including crash gaps.
        with self.db() as db:
            for p in (self.root / "jobs").iterdir():
                if not (p/'status.json').is_file():continue
                r = read_json(p / "status.json")
                if r["status"] == "RUNNING":
                    r["status"] = "INTERRUPTED"
                    write_json(p / "status.json", r)
                    self.seal(r['id'])
                db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?)", (r["id"], r["status"], r["created"]))

    def next_job(self):
        with self.db() as db:
            row = db.execute("SELECT id FROM jobs WHERE status='QUEUED' ORDER BY created LIMIT 1").fetchone()
        return row[0] if row else None

    def seal(self,id):
        directory=self.path("jobs",id)
        files={p.relative_to(directory).as_posix():{"sha256":digest(p),"bytes":p.stat().st_size}
               for p in directory.rglob("*") if p.is_file() and p.name not in {"status.json","manifest.json"} and ".tmp-" not in p.name}
        write_json(directory/"manifest.json",files)
