"""Host-side baseline: versions, image identity, font volume and renderer profile.

The service can report its own code version and font manifest, but only the host
knows the image digest, the mounted font bytes and the core compiler commit.
This report is deliberately separate from task output: it is evidence for a
deployment, not a per-job acceptance record. Every command is optional and a
failure is recorded as unknown instead of aborting the report.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def _run(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unknown", "error": type(error).__name__}
    if result.returncode != 0:
        return {"status": "unknown", "returncode": result.returncode,
                "error": result.stderr.strip()[:500]}
    return {"status": "ok", "stdout": result.stdout.strip()}


def _git(root):
    if not root:
        return {"status": "unknown", "reason": "not_configured"}
    root = Path(root)
    if not (root / ".git").exists():
        return {"status": "unknown", "reason": "not_a_git_checkout"}
    return {
        "root": str(root.resolve()),
        "commit": _run(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "branch": _run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _run(["git", "-C", str(root), "status", "--porcelain"]),
    }


def _image(image):
    if not image:
        return {"status": "unknown", "reason": "not_configured"}
    inspected = _run(["docker", "image", "inspect", image])
    report = {"status": inspected["status"], "reference": image}
    if inspected["status"] == "ok":
        try:
            item = json.loads(inspected["stdout"])[0]
            report.update(id=item.get("Id"), repo_digests=item.get("RepoDigests", []),
                          created=item.get("Created"))
        except (ValueError, KeyError, IndexError):
            report.update(status="unknown", reason="unreadable_image_inspect")
    else:
        report.update(reason=inspected.get("error"))
    return report


def _font_volume(manifest):
    if not manifest:
        return {"status": "unknown", "reason": "not_configured"}
    path = Path(manifest).expanduser()
    report = {"configured": True, "exists": path.is_file()}
    if not path.is_file():
        report.update(status="unavailable", reason="manifest_missing")
        return report
    raw = path.read_bytes()
    report.update(status="ready", manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  manifest_bytes=len(raw))
    try:
        document = json.loads(raw)
        report["declared_fonts"] = len(document.get("fonts", []))
    except (ValueError, AttributeError):
        report.update(status="degraded", reason="manifest_invalid_json")
        return report
    files = []
    for entry in document.get("fonts", []) if isinstance(document, dict) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        target = (path.parent / entry["path"]).resolve()
        try:
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            files.append({"family": entry.get("family"), "bold": entry.get("bold"),
                          "sha256": digest, "bytes": target.stat().st_size})
        except OSError:
            files.append({"family": entry.get("family"), "bold": entry.get("bold"),
                          "status": "missing"})
    report["fonts"] = files
    return report


def main():
    parser = argparse.ArgumentParser(description="Write an image2ppt-vision deployment baseline.")
    parser.add_argument("--vision-root", help="vision development checkout")
    parser.add_argument("--core-root", help="image2ppt-core checkout")
    parser.add_argument("--image", help="Docker image reference, e.g. image2ppt-vision:0.3.0")
    parser.add_argument("--font-manifest", help="VISION_FONT_MANIFEST file")
    parser.add_argument("--output", default="environment-baseline.json")
    args = parser.parse_args()
    report = {
        "schema": "Image2PPTVisionEnvironment/1",
        "vision": _git(args.vision_root),
        "core": _git(args.core_root),
        "image": _image(args.image),
        "font_manifest": _font_volume(args.font_manifest),
        "tools": {
            "python": _run(["python", "--version"]),
            "node": _run(["node", "--version"]),
            "docker": _run(["docker", "--version"]),
            "docker_present": bool(shutil.which("docker")),
        },
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "status": "degraded" if report["font_manifest"].get("status") != "ready"
                      else "ready"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
