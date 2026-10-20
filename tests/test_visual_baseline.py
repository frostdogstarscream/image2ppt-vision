"""A frozen regression reference must detect changed or newly inserted artifacts."""
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/freeze_visual_baseline.py"
spec = importlib.util.spec_from_file_location("visual_baseline", SCRIPT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def snapshot(tmp_path):
    artifact = tmp_path / "retained.bin"
    artifact.write_bytes(b"original")
    baseline.write_json(tmp_path / "manifest.json", {
        "pages": [], "files": [{"path": "retained.bin", "sha256": baseline.digest(artifact)}]})
    return artifact


def test_changed_artifact_rejected(tmp_path):
    artifact = snapshot(tmp_path)
    assert baseline.verify(tmp_path)["status"] == "PASS"
    artifact.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        baseline.verify(tmp_path)


def test_extra_nested_manifest_not_hidden(tmp_path):
    snapshot(tmp_path)
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        baseline.verify(tmp_path)


def test_manifest_cannot_reference_outside_snapshot(tmp_path):
    baseline.write_json(tmp_path / "manifest.json", {
        "pages": [], "files": [{"path": "../outside.bin", "sha256": "anything"}]})
    with pytest.raises(ValueError, match="escapes"):
        baseline.verify(tmp_path)
