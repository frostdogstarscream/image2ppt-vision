"""Synthetic checks establish mechanics, not real-slide or Office acceptance."""
from pathlib import Path
import hashlib
import json
import socket
import urllib.request

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from image2ppt_vision.typography import estimate_text, font_status


@pytest.fixture
def candidate():
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            font = ImageFont.truetype(path, 32)
            return {"path": path, "family": font.getname()[0], "bold": False}
    pytest.skip("No test font installed")


def render(candidate, text="Actual text 2026", size=32, bg=(245, 246, 250), fg=(25, 40, 65), gradient=False):
    font = ImageFont.truetype(candidate["path"], size)
    width = int(font.getlength(text)) + 100
    rgb = np.broadcast_to(np.array(bg, np.uint8), (110, width, 3)).copy()
    if gradient:
        rgb = np.clip(rgb.astype(float) + np.linspace(-15, 0, width)[None, :, None], 0, 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    ImageDraw.Draw(im).text((35, 20), text, font=font, fill=fg)
    bounds = font.getbbox(text)
    ink = [35+bounds[0], 20+bounds[1], bounds[2]-bounds[0], bounds[3]-bounds[1]]
    # Deliberately inaccurate detector padding must not inflate the font.
    bbox = [ink[0]-7, ink[1]-9, ink[2]+14, ink[3]+18]
    return np.asarray(im), rgb, {"text": text, "bbox": bbox}, ink


@pytest.mark.parametrize("bg,fg,gradient", [
    ((245, 246, 250), (25, 40, 65), False),
    ((15, 35, 65), (245, 248, 252), False),
    ((235, 240, 250), (25, 50, 110), True),
    ((40, 65, 95), (240, 248, 255), True),
])
def test_font_size_uses_ink_and_cleans_dark_light_gradient(candidate, bg, fg, gradient):
    image, background, ocr, ink = render(candidate, bg=bg, fg=fg, gradient=gradient)
    original = image.copy()
    result = estimate_text(image, ocr, [candidate])
    assert result["native_eligible"], result["diagnostics"]
    assert abs(result["font_size_px"]-32) <= 1
    assert abs(result["bbox"][1]-ink[1]) <= 1
    x, y, w, h = result["patch_bbox"]
    assert np.mean(np.abs(result["cleaned_patch"].astype(float)-background[y:y+h, x:x+w])) < 1
    assert result["color"] == "#"+"".join(f"{v:02X}" for v in fg)
    np.testing.assert_array_equal(image, original)
    # Broad OCR padding outside actual ink is preserved exactly.
    np.testing.assert_array_equal(result["cleaned_patch"][:4], image[y:y+4, x:x+w])


def test_photo_background_is_not_erased(candidate):
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
    result = estimate_text(image, {"text": "Picture text", "bbox": [20, 20, 250, 50]}, [candidate])
    assert not result["native_eligible"]
    assert result["cleaned_patch"] is None
    assert result["diagnostics"]["reason"] == "non_smooth_or_crossing_background"


def test_graphic_crossing_is_not_painted(candidate):
    image, _, ocr, _ = render(candidate)
    image = image.copy()
    x, y, w, h = map(int, ocr["bbox"])
    image[y+h-4:y+h-2, x+3:x+w-3] = (20, 120, 180)
    result = estimate_text(image, ocr, [candidate])
    assert not result["native_eligible"]
    assert result["cleaned_patch"] is None


def test_multicolor_requires_character_alignment(candidate):
    font = ImageFont.truetype(candidate["path"], 32)
    text = "Native COLOR text"
    image, _, ocr, _ = render(candidate, text=text)
    im = Image.fromarray(np.full_like(image, 250))
    draw = ImageDraw.Draw(im)
    x = 35
    for part, color in (("Native ", (20, 30, 45)), ("COLOR", (210, 20, 40)), (" text", (20, 30, 45))):
        draw.text((x, 20), part, font=font, fill=color)
        x += font.getlength(part)
    result = estimate_text(np.asarray(im), ocr, [candidate])
    assert result["native_eligible"], result["diagnostics"]
    assert "".join(r["text"] for r in result["runs"]) == text
    assert len({r["color"] for r in result["runs"]}) == 2


def test_wrong_ocr_preserves_original(candidate):
    image, _, ocr, _ = render(candidate)
    ocr["text"] = "xxxxxxxxxxxxxxxxxxx"
    result = estimate_text(image, ocr, [candidate])
    assert not result["native_eligible"]
    assert result["cleaned_patch"] is None


def test_missing_font_and_invalid_image(candidate):
    image, _, ocr, _ = render(candidate)
    result = estimate_text(image, ocr, [])
    assert result["diagnostics"]["reason"] == "no_usable_installed_font"
    with pytest.raises(ValueError, match="uint8"):
        estimate_text(image.astype(float), ocr, [candidate])


@pytest.mark.parametrize("bold", [False, True])
def test_resampled_text_keeps_weight_and_natural_size(bold):
    regular, heavy = Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyhbd.ttc")
    if not regular.is_file() or not heavy.is_file():
        pytest.skip("Windows CJK font pair unavailable")
    candidates = [{"path": str(regular), "family": "Microsoft YaHei", "bold": False},
                  {"path": str(heavy), "family": "Microsoft YaHei", "bold": True}]
    font = ImageFont.truetype(str(heavy if bold else regular), 64)
    text = "业务咨询与条文定位需求高"
    im = Image.new("RGB", (1000, 180), "#F2F7FD")
    ImageDraw.Draw(im).text((40, 20), text, font=font, fill="#263752")
    im = im.resize((500, 90), Image.Resampling.LANCZOS)
    bounds = font.getbbox(text)
    ocr = {"text": text, "bbox": [16, 6+bounds[1]/2, (bounds[2]-bounds[0])/2+8,
                                     (bounds[3]-bounds[1])/2+8]}
    r = estimate_text(np.asarray(im), ocr, candidates)
    assert r["native_eligible"], r["diagnostics"]
    assert r["bold"] is bold
    assert abs(r["font_size_px"]-32) <= 1


@pytest.fixture
def font_manifest(tmp_path, monkeypatch):
    # This tests manifest validation without copying/distributing system fonts.
    # Font file parsing is mocked only in the successful manifest tests.
    data = b"synthetic-local-font-test-asset"
    asset = tmp_path / "fixture.font"
    asset.write_bytes(data)
    manifest = tmp_path / "fonts.json"
    entry = {"path": asset.name, "family": "Private Test Family", "bold": False,
             "sha256": hashlib.sha256(data).hexdigest()}
    manifest.write_text(json.dumps({"fonts": [entry]}), encoding="utf8")
    monkeypatch.setenv("VISION_FONT_MANIFEST", str(manifest))
    return manifest, asset, entry


def test_font_manifest_status_is_sanitized_and_never_downloads(font_manifest, monkeypatch):
    manifest, asset, _ = font_manifest

    def unexpected_network(*args, **kwargs):
        raise AssertionError("font configuration must never access the network")

    monkeypatch.setattr(socket, "create_connection", unexpected_network)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_network)
    monkeypatch.setattr(ImageFont, "truetype", lambda *args, **kwargs: object())
    status = font_status()
    assert status == {"source": "configured_manifest", "font_count": 1,
                      "families": ["Private Test Family"],
                      "hash_verification": "passed", "sha256_verified": True}
    assert str(manifest.parent) not in json.dumps(status)
    assert asset.name not in json.dumps(status)


def test_manifest_changed_font_invalidates_hash_cache(font_manifest, monkeypatch):
    _, asset, _ = font_manifest
    monkeypatch.setattr(ImageFont, "truetype", lambda *args, **kwargs: object())
    assert font_status()["sha256_verified"]
    asset.write_bytes(b"changed font asset contents")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        font_status()


@pytest.mark.parametrize("bad_path", ["../outside.ttf", "sub/../../outside.ttf",
                                     "C:/Windows/Fonts/msyh.ttc", "/tmp/font.ttf",
                                     "https://example.invalid/font.ttf"])
def test_manifest_rejects_unconfined_paths(font_manifest, bad_path):
    manifest, _, entry = font_manifest
    entry["path"] = bad_path
    manifest.write_text(json.dumps({"fonts": [entry]}), encoding="utf8")
    with pytest.raises(ValueError, match="relative|confined|traverse"):
        font_status()


def test_configured_missing_corrupt_or_unloadable_fonts_never_fall_back(font_manifest):
    manifest, asset, entry = font_manifest
    # Even a hash-correct asset must be a loadable font.
    with pytest.raises(ValueError, match="cannot be loaded"):
        font_status()
    asset.unlink()
    with pytest.raises(ValueError, match="missing or unreadable"):
        font_status()
    manifest.write_text("{broken", encoding="utf8")
    with pytest.raises(ValueError, match="invalid JSON"):
        font_status()
    manifest.unlink()
    with pytest.raises(ValueError, match="unreadable"):
        font_status()


def test_manifest_requires_hash_and_limits_font_count(font_manifest):
    manifest, _, entry = font_manifest
    manifest.write_text(json.dumps({"fonts": [entry]*33}), encoding="utf8")
    with pytest.raises(ValueError, match="1 to 32"):
        font_status()
    del entry["sha256"]
    manifest.write_text(json.dumps({"fonts": [entry]}), encoding="utf8")
    with pytest.raises(ValueError, match="SHA256"):
        font_status()


def test_remote_manifest_is_rejected_without_network(monkeypatch):
    monkeypatch.setenv("VISION_FONT_MANIFEST", "https://example.invalid/fonts.json")
    with pytest.raises(ValueError, match="local file"):
        font_status()


def test_resampled_text_deletion_removes_low_opacity_glyph_ghosts(candidate):
    # Lanczos produces several layers of extremely pale AA pixels. Deleting
    # only pixels over the detection threshold leaves a legible second copy.
    font = ImageFont.truetype(candidate["path"], 32*4)
    text = "AA text"
    large = Image.new("RGB", (1600, 400), (245, 247, 250))
    ImageDraw.Draw(large).text((80, 40), text, font=font, fill=(20, 45, 75))
    image = np.asarray(large.resize((400, 100), Image.Resampling.LANCZOS))
    changed = np.max(np.abs(image.astype(float)-[245,247,250]), axis=2)
    yy, xx = np.where(changed > 8)
    ocr = {"text": text, "bbox": [int(xx.min())-5,int(yy.min())-5,
                                   int(xx.max()-xx.min())+11,int(yy.max()-yy.min())+11]}
    result = estimate_text(image, ocr, [candidate])
    assert result["native_eligible"], result["diagnostics"]
    # Independent known clean background, not the estimator's own fitted model.
    error = np.abs(result["cleaned_patch"].astype(float)-[245,247,250])
    assert error.max() <= 1, (error.max(), result["diagnostics"])
    assert int((error.max(axis=2) > 1).sum()) == 0


def test_cleanup_preserves_adjacent_line_outside_owned_text(candidate):
    font = ImageFont.truetype(candidate["path"], 24)
    image = Image.new("RGB", (360, 110), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 8), "First line", font=font, fill="#142744")
    draw.text((25, 46), "Keep this line", font=font, fill="#142744")
    rgb = np.asarray(image)
    bounds = font.getbbox("First line")
    result = estimate_text(rgb, {"text": "First line", "bbox": [22,8+bounds[1]-3,
                           bounds[2]-bounds[0]+6,bounds[3]-bounds[1]+6]}, [candidate])
    assert result["native_eligible"], result["diagnostics"]
    reconstructed = rgb.copy()
    x,y,w,h = result["patch_bbox"]
    reconstructed[y:y+h,x:x+w] = result["cleaned_patch"]
    np.testing.assert_array_equal(reconstructed[46:], rgb[46:])
