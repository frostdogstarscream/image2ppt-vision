"""One-command, saved-evidence deck conversion through the fixed compiler.

All input revisions are snapshotted. This module orchestrates the independent
reconstruction layer and a known local compose.mjs; model output is never code.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .plans import load_plan
from .reconstruction import reconstruct
from .storage import digest, read_json, write_json


MAX_COMPILER_OUTPUT_BYTES = 4_000_000
MAX_PAGES = 200
REQUIRED_COMPILER_FILES = (
    "scripts/compose.mjs", "scripts/compile.mjs", "scripts/runtime.mjs", "scripts/ir.mjs", "scripts/pptx_xml.py",
)


class ConversionError(RuntimeError):
    """A stable code; compiler output and source content are never in the message."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _directory(value, code):
    path = Path(value).resolve()
    if not path.is_dir():
        raise ConversionError(code)
    return path


def _files(directory, suffix):
    files = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == suffix), key=lambda p: p.name)
    folded = [p.stem.casefold() for p in files]
    if len(folded) != len(set(folded)):
        raise ConversionError("duplicate_input_stems")
    return files


def _collect(input_dir, analysis_dir, plan_dir):
    source_dir = _directory(input_dir, "input_directory_missing")
    evidence_dir = _directory(analysis_dir, "analysis_directory_missing")
    plans_dir = _directory(plan_dir, "plan_directory_missing")
    sources = _files(source_dir, ".png")
    analyses = {p.stem: p for p in _files(evidence_dir, ".json")}
    plans = {p.stem: p for p in _files(plans_dir, ".json")}
    if not sources or len(sources) > MAX_PAGES:
        raise ConversionError("invalid_page_count")
    stems = {p.stem for p in sources}
    if set(analyses) != stems:
        raise ConversionError("analysis_stems_mismatch")
    if set(plans) != stems:
        raise ConversionError("plan_stems_mismatch")
    return [(source, analyses[source.stem], plans[source.stem]) for source in sources]


def _compiler(compiler_root):
    root = _directory(compiler_root, "compiler_root_missing")
    if any(not (root / name).is_file() or not (root / name).resolve().is_relative_to(root)
           for name in REQUIRED_COMPILER_FILES):
        raise ConversionError("fixed_compiler_files_missing")
    configured = os.environ.get("VISION_NODE")
    found = shutil.which(configured or "node")
    if not found and configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            found = str(candidate.resolve())
    if not found:
        raise ConversionError("node_executable_missing")
    node = Path(found).resolve()
    if not node.is_file() or node.suffix.lower() in {".cmd", ".bat", ".ps1", ".sh"}:
        raise ConversionError("invalid_node_executable")
    return root, node


def _bytes(value):
    return value if isinstance(value, bytes) else (value or "").encode("utf-8", errors="replace")


def _diagnostics(stdout, stderr):
    result = {}
    for name, value in (("stdout", stdout), ("stderr", stderr)):
        data = _bytes(value)
        result[name + "_bytes"] = len(data)
        result[name + "_sha256"] = hashlib.sha256(data).hexdigest()
    return result


def _final_json(stdout):
    data = _bytes(stdout)
    if len(data) > MAX_COMPILER_OUTPUT_BYTES:
        raise ConversionError("compiler_output_too_large")
    text = data.decode("utf-8", errors="replace").lstrip("\ufeff")
    decoder = json.JSONDecoder()
    # The fixed compiler prints a multiline JSON object after any diagnostics.
    # Inspect line starts only, require the object to consume the entire tail.
    starts = [match.start(1) for match in re.finditer(r"(?:^|\n)[ \t]*(\{)", text)]
    for start in reversed(starts[-1000:]):
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict) and not text[end:].strip():
            return value
    raise ConversionError("compiler_result_json_missing")


def _artifact_path(value, parent, *, directory=False):
    if not isinstance(value, str) or not value or "://" in value:
        raise ConversionError("invalid_compiler_artifact_path")
    path = Path(value)
    if not path.is_absolute():
        raise ConversionError("invalid_compiler_artifact_path")
    path = path.resolve()
    if not path.is_relative_to(parent) or path == parent or not (path.is_dir() if directory else path.is_file()):
        raise ConversionError("compiler_artifact_outside_output_or_missing")
    return path


def _compiled_artifacts(result, build_dir, page_count):
    if type(result.get("slide_count")) is not int or result["slide_count"] != page_count:
        raise ConversionError("compiler_slide_count_mismatch")
    output_dir = _artifact_path(result.get("output_dir"), build_dir, directory=True)
    pptx = _artifact_path(result.get("pptx"), output_dir)
    quality_path = _artifact_path(result.get("quality"), output_dir)
    preview = _artifact_path(result.get("preview"), output_dir, directory=True)
    if pptx.suffix.lower() != ".pptx":
        raise ConversionError("invalid_compiler_pptx")
    quality = read_json(quality_path)
    pptx_sha256 = digest(pptx)
    if quality.get("slide_count") != page_count or quality.get("pptx_sha256") != pptx_sha256:
        raise ConversionError("compiler_quality_mismatch")
    if quality.get("preview", {}).get("origin") != "reimported_final_pptx":
        raise ConversionError("compiler_preview_origin_missing")
    preview_files = sorted(p for p in preview.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if len(preview_files) != page_count:
        raise ConversionError("compiler_preview_count_mismatch")
    return {"output_dir": str(output_dir), "pptx": str(pptx), "pptx_sha256": pptx_sha256,
            "quality": str(quality_path), "preview": str(preview), "slide_count": page_count,
            "preview_origin": "reimported_final_pptx"}


def convert(input_dir, analysis_dir, plan_dir, compiler_root, output, *, compile_timeout=1800, layers_dir=None):
    """Reconstruct sorted PNG pages, then compose with the fixed local compiler.

    The output directory must be new. conversion.json records partial progress
    and sanitized failure diagnostics. Existing inputs/output are never removed.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("conversion output must be a new directory")
    if isinstance(compile_timeout, bool) or not isinstance(compile_timeout, (int, float)) or not 0 < compile_timeout <= 7200:
        raise ConversionError("invalid_compile_timeout")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "DeckConversion", "version": "1.0", "engine_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING", "stage": "preflight",
                "pages": [], "compiler": None, "human_acceptance": False, "powerpoint_checked": False, "wps_checked": False}
    manifest_path = output / "conversion.json"
    write_json(manifest_path, manifest)
    try:
        inputs = _collect(input_dir, analysis_dir, plan_dir)
        layers_by_stem={}
        if layers_dir is not None:
            layers_by_stem={p.stem:p for p in _files(_directory(layers_dir,'layers_directory_missing'),'.json')}
            if set(layers_by_stem)!={s.stem for s,_,_ in inputs}:
                raise ConversionError('layer_stems_mismatch')
        compiler, node = _compiler(compiler_root)
        manifest["source_order"] = [source.name for source, _, _ in inputs]
        input_root = output / "inputs"
        for name in ("source", "analysis", "plans"):
            (input_root / name).mkdir(parents=True)
        irs = []
        manifest["stage"] = "reconstruction"
        for source, analysis_path, plan_path in inputs:
            row = {"source": source.name, "source_sha256": digest(source), "analysis_sha256": digest(analysis_path),
                   "plan_sha256": digest(plan_path), "status": "RUNNING", "output": "pages/" + source.stem}
            manifest["pages"].append(row); write_json(manifest_path, manifest)
            source_copy = input_root / "source" / source.name
            analysis_copy = input_root / "analysis" / analysis_path.name
            plan_copy = input_root / "plans" / plan_path.name
            for original, copied, expected in ((source,source_copy,row["source_sha256"]),
                                              (analysis_path,analysis_copy,row["analysis_sha256"]),
                                              (plan_path,plan_copy,row["plan_sha256"])):
                shutil.copyfile(original, copied)
                if digest(copied) != expected:
                    raise ConversionError("input_changed_during_snapshot")
            page_output = output / "pages" / source.stem
            plan = load_plan(plan_copy, source_sha256=row["source_sha256"])
            options={}
            if layers_dir is not None:
                from .layers import snapshot_layers
                evidence=read_json(analysis_copy)
                saved=snapshot_layers(layers_by_stem[source.stem],input_root/'layers'/source.stem,
                        source_sha256=row['source_sha256'],image_size=evidence['image_size'],
                        texts=[t for t in evidence['objects'] if t['kind']=='text'])
                row['layers_sha256']=digest(saved);options['layers']=saved
            report = reconstruct(source_copy, read_json(analysis_copy), plan, page_output,**options)
            ir = page_output / "slideir.json"
            if not ir.is_file() or not (page_output / "authoring.json").is_file():
                raise ConversionError("reconstruction_artifacts_missing")
            row.update({"status": "REVIEW_REQUIRED", "ir": ir.relative_to(output).as_posix(), "ir_sha256": digest(ir),
                        **{key:report[key] for key in ("ocr_lines","native_text","native_text_ratio","semantic_regions",
                                  "native_module_backdrops","independent_images","groups","region_residual_fraction") if key in report}})
            irs.append(ir); write_json(manifest_path, manifest)
        manifest["stage"] = "compile"
        build_dir = output / "build"; build_dir.mkdir()
        command = [str(node), str(compiler / "scripts/compose.mjs"), *map(str, irs), "--output", str(build_dir)]
        if os.name == "nt" and len(subprocess.list2cmdline(command)) > 30000:
            raise ConversionError("compiler_argument_limit")
        manifest["compiler"] = {"program": "fixed_compose_mjs", "compose_sha256": digest(compiler / "scripts/compose.mjs"),
                               "timeout_seconds": compile_timeout, "shell": False}
        write_json(manifest_path, manifest)
        try:
            completed = subprocess.run(command, cwd=str(compiler), shell=False, check=False, capture_output=True,
                                       encoding="utf-8", errors="replace", timeout=compile_timeout,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.TimeoutExpired as exc:
            manifest["compiler"].update(_diagnostics(exc.output, exc.stderr))
            raise ConversionError("compiler_timeout") from None
        manifest["compiler"].update(returncode=completed.returncode, **_diagnostics(completed.stdout,completed.stderr))
        if completed.returncode != 0:
            raise ConversionError("compiler_failed")
        artifacts = _compiled_artifacts(_final_json(completed.stdout), build_dir, len(inputs))
        manifest["artifacts"] = {key: (str(Path(value).relative_to(output).as_posix()) if key in
                                      {"output_dir","pptx","quality","preview"} else value) for key,value in artifacts.items()}
        manifest["status"] = "REVIEW_REQUIRED"; manifest["stage"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        return {"status":"REVIEW_REQUIRED",**artifacts,"build_output_dir":artifacts["output_dir"],
                "output_dir":str(output),"conversion":str(manifest_path),
                "human_acceptance":False,"powerpoint_checked":False,"wps_checked":False}
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"; write_json(manifest_path,manifest)
        raise
    except Exception as exc:
        code = exc.code if isinstance(exc, ConversionError) else "conversion_" + manifest["stage"] + "_failed"
        manifest["status"] = "FAILED"
        manifest["error"] = {"code":code,"type":type(exc).__name__}
        if manifest["pages"] and manifest["pages"][-1]["status"] == "RUNNING":
            manifest["pages"][-1]["status"] = "FAILED"
        write_json(manifest_path,manifest)
        raise ConversionError(code) from None
