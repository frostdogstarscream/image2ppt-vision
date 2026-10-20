"""Plan/provider contract checks; no live provider calls or accuracy claims."""
import copy
import hashlib
import io
import json
from urllib import error
from unittest.mock import Mock

import pytest
from PIL import Image

from image2ppt_vision.plans import (
    MAX_PLAN_BYTES, PlanValidationError, load_plan, parse_plan_json, validate_plan,
)
from image2ppt_vision import model_provider as provider


def example_plan(source_hash="a" * 64):
    return {
        "schema": "ReconstructionPlan", "version": "1.0", "source_sha256": source_hash,
        "image_size": [200, 100], "author": {"tool": "vision-model-plan", "model_id": "test-model"},
        "regions": [
            {"id": "header", "bbox": [0, 0, 200, 20], "kind": "header"},
            {"id": "content", "bbox": [0, 20, 200, 80], "kind": "panel", "label": "正文区"},
        ],
        "text_overrides": [{"target_id": "text-0000", "text": "核对后的文字", "reason": "原图可见"}],
        "shapes": [{"id": "card", "bbox": [10, 30, 180, 60], "geometry": "roundRect", "fill": "224488",
                    "corner_radius_px": 5, "line": {"color": "FFFFFF", "width_px": 1.5}}],
    }


def test_valid_plan_round_trip_and_ocr_binding(tmp_path):
    data = example_plan()
    plan = validate_plan(data, source_sha256="a" * 64, image_size=[200, 100], ocr_ids={"text-0000"})
    assert plan.regions[0].bbox[3] == plan.regions[1].bbox[1]
    assert plan.shapes[0].native_only_if_verified is True
    assert plan.to_dict()["schema"] == "ReconstructionPlan"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False), encoding="utf-8")
    assert load_plan(path, ocr_ids={"text-0000"}).to_dict() == plan.to_dict()
    with pytest.raises(PlanValidationError, match="unknown OCR"):
        load_plan(path, ocr_ids=set())


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(source_sha256="A" * 64),
    lambda p: p.update(source_sha256="a" * 63),
    lambda p: p.update(image_size=[True, 100]),
    lambda p: p.update(image_size=["200", 100]),
    lambda p: p.update(image_size=[0, 100]),
    lambda p: p.update(image_size=[40001, 1000]),
    lambda p: p.update(version="2.0"),
    lambda p: p.update(schema="SlideIR"),
    lambda p: p.update(regions=[]),
    lambda p: p["regions"][0].update(bbox=[0, 0, 201, 20]),
    lambda p: p["regions"][0].update(bbox=[0, 0, 0, 20]),
    lambda p: p["regions"][0].update(bbox=[-1, 0, 200, 20]),
    lambda p: p["regions"][0].update(bbox=[0, 0, float("nan"), 20]),
    lambda p: p["regions"][0].update(bbox=[0, 0, float("inf"), 20]),
    lambda p: p["regions"][0].update(bbox=[False, 0, 200, 20]),
    lambda p: p["regions"][0].update(bbox=[0, 0, "200", 20]),
    lambda p: p["regions"][1].update(bbox=[0, 19.999, 200, 80]),
    lambda p: p["regions"][1].update(id="header"),
    lambda p: p["shapes"][0].update(id="header"),
    lambda p: p["regions"][0].update(id="../../file"),
    lambda p: p["regions"][0].update(kind="script"),
    lambda p: p["regions"][0].update(asset="https://example.test/image.png"),
    lambda p: p.update(code="raise SystemExit()"),
    lambda p: p["shapes"][0].update(fill="url(https://example.test)"),
    lambda p: p["shapes"][0].update(geometry="customPath"),
    lambda p: p["shapes"][0].update(corner_radius_px=40),
    lambda p: p["shapes"][0].update(geometry="rect"),
    lambda p: p["shapes"][0].update(native_only_if_verified=False),
    lambda p: p["shapes"][0].update(native_only_if_verified=1),
    lambda p: p["shapes"][0]["line"].update(width_px=0),
    lambda p: p["text_overrides"].append(copy.deepcopy(p["text_overrides"][0])),
    lambda p: p["text_overrides"][0].update(text=""),
    lambda p: p["text_overrides"][0].update(reason="x" * 1001),
])
def test_invalid_plan_is_rejected(mutation):
    data = example_plan()
    mutation(data)
    with pytest.raises(PlanValidationError):
        validate_plan(data)


def test_context_binding_and_mutation_are_rechecked():
    data = example_plan()
    with pytest.raises(PlanValidationError, match="hash mismatch"):
        validate_plan(data, source_sha256="b" * 64)
    with pytest.raises(PlanValidationError, match="dimensions mismatch"):
        validate_plan(data, image_size=[201, 100])
    plan = validate_plan(data)
    plan.regions[0].bbox[2] = 500
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


@pytest.mark.parametrize("content", [
    '```json\n{}\n```', '{"regions": [], "regions": []}', '[]', '{"x":',
    'x' * (MAX_PLAN_BYTES + 1), b'\xff\xff',
], ids=["markdown", "duplicate-key", "array", "malformed", "oversized", "invalid-utf8"])
def test_raw_json_is_bounded_and_unambiguous(content):
    with pytest.raises(PlanValidationError):
        parse_plan_json(content)


@pytest.fixture
def source(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image)
    source_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    analysis = {"image_size": [200, 100], "source_sha256": source_hash, "objects": [
        {"id": "text-0000", "kind": "text", "text": "私有源文本", "bbox": [15, 35, 100, 25]},
        {"id": "shape-0000", "kind": "rectangle", "bbox": [10, 30, 180, 60]},
    ]}
    return image, analysis, example_plan(source_hash)


@pytest.fixture
def configuration(monkeypatch):
    monkeypatch.setenv("VISION_LLM_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("VISION_LLM_MODEL", "test-model")
    monkeypatch.setenv("VISION_LLM_API_KEY", "secret-fixture-key")


def completion(data, **choice_options):
    return json.dumps({"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(data, ensure_ascii=False)}, **choice_options}]}).encode("utf-8")


def fake_transport(monkeypatch, data):
    opener = Mock()
    response = io.BytesIO(data)
    opener.open.return_value = response
    builder = Mock(return_value=opener)
    monkeypatch.setattr(provider.request, "build_opener", builder)
    return opener, builder


def test_provider_explicit_request_and_hash_only_provenance(source, configuration, monkeypatch, tmp_path):
    image, analysis, data = source
    opener, builder = fake_transport(monkeypatch, completion(data))
    # Configuration and module imports have not invoked any model operation.
    assert not builder.called
    provenance_path = tmp_path / "provenance.json"
    result = provider.generate_plan(image, analysis, provenance_path=provenance_path)
    assert result.source_sha256 == data["source_sha256"]
    req = opener.open.call_args.args[0]
    assert req.full_url == "https://provider.example/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer secret-fixture-key"
    payload = json.loads(req.data)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "私有源文本" in payload["messages"][1]["content"][0]["text"]
    saved = provenance_path.read_text("utf-8")
    for private in ("secret-fixture-key", "私有源文本", "核对后的文字", "provider.example", "data:image"):
        assert private not in saved
    provenance = json.loads(saved)
    assert provenance["human_acceptance"] is False
    assert provenance["attempts"] == 1
    assert len(provenance["response_sha256"]) == 64
    assert isinstance(builder.call_args.args[0], provider._NoRedirect)


@pytest.mark.parametrize("endpoint", [
    "http://example.test/v1", "file:///tmp/test", "https://user:pass@example.test/v1",
    "https://example.test/v1?key=secret", "https://example.test/v1#secret", "https://example.test:99999/v1",
    "https://example.test/\nsecret", "https://", "ftp://127.0.0.1/v1",
])
def test_unsafe_endpoints_are_rejected(endpoint):
    with pytest.raises(provider.ModelProviderError, match="invalid_model_endpoint"):
        provider._endpoint(endpoint)


@pytest.mark.parametrize("endpoint", [
    "http://127.0.0.1:8000/v1", "http://localhost:8000/v1", "http://[::1]:8000/v1",
    "https://example.test/v1", "https://example.test/v1/chat/completions/",
])
def test_safe_endpoints(endpoint):
    assert provider._endpoint(endpoint).endswith("/chat/completions")


def test_redirect_rejected_without_credentials_or_url_in_error():
    with pytest.raises(provider.ModelProviderError) as raised:
        provider._NoRedirect().redirect_request(None, None, 302, "secret", {}, "https://secret.example")
    assert str(raised.value) == "model_redirect_rejected"


def test_trickled_response_is_time_bounded(monkeypatch):
    response = Mock()
    response.read1.return_value = b"x"
    monkeypatch.setattr(provider.time, "monotonic", Mock(side_effect=[0, 2]))
    with pytest.raises(provider.ModelProviderError, match="model_response_timeout"):
        provider._read_response(response, 1)
    assert response.read1.call_count == 1


@pytest.mark.parametrize("status,expected_calls", [(400, 1), (401, 1), (429, 2), (500, 2)])
def test_http_errors_are_bounded_and_sanitized(source, configuration, monkeypatch, status, expected_calls):
    image, analysis, _ = source
    opener, _ = fake_transport(monkeypatch, b"")
    opener.open.side_effect = [
        error.HTTPError("https://secret.example", status, "secret response", {}, io.BytesIO(b"private-body"))
        for _ in range(2)
    ]
    with pytest.raises(provider.ModelProviderError) as raised:
        provider.generate_plan(image, analysis)
    assert str(raised.value) == "model_http_" + str(status)
    assert opener.open.call_count == expected_calls


def test_retry_can_recover_transport_error(source, configuration, monkeypatch):
    image, analysis, data = source
    opener, _ = fake_transport(monkeypatch, b"")
    opener.open.side_effect = [error.URLError("secret transport detail"), io.BytesIO(completion(data))]
    assert provider.generate_plan(image, analysis).source_sha256 == data["source_sha256"]
    assert opener.open.call_count == 2


@pytest.mark.parametrize("response_kind", ["truncated", "code", "unknown_ocr", "wrong_hash", "wrong_author", "oversized", "bad_json", "tool_call"])
def test_model_output_cannot_escape_contract(source, configuration, monkeypatch, response_kind):
    image, analysis, data = source
    if response_kind == "truncated":
        body = completion(data, finish_reason="length")
    elif response_kind == "code":
        data["code"] = "__import__('os').system('secret-command')"
        body = completion(data)
    elif response_kind == "unknown_ocr":
        data["text_overrides"][0]["target_id"] = "fabricated"
        body = completion(data)
    elif response_kind == "wrong_hash":
        data["source_sha256"] = "b" * 64
        body = completion(data)
    elif response_kind == "wrong_author":
        data["author"]["model_id"] = "unrequested-model"
        body = completion(data)
    elif response_kind == "oversized":
        body = b"x" * (provider.MAX_RESPONSE_BYTES + 1)
    elif response_kind == "tool_call":
        body = completion(data, message={"content": json.dumps(data), "tool_calls": [{"function": "secret"}]})
    else:
        body = b"private invalid response body"
    opener, _ = fake_transport(monkeypatch, body)
    with pytest.raises(provider.ModelProviderError) as raised:
        provider.generate_plan(image, analysis)
    assert "secret" not in str(raised.value) and "private" not in str(raised.value)
    # Invalid model responses are not retried or silently repaired.
    assert opener.open.call_count == 1


def test_ocr_source_binding_prevents_network_call(source, configuration, monkeypatch):
    image, analysis, data = source
    opener, _ = fake_transport(monkeypatch, completion(data))
    analysis["source_sha256"] = "b" * 64
    with pytest.raises(provider.ModelProviderError, match="analysis_source_mismatch"):
        provider.generate_plan(image, analysis)
    opener.open.assert_not_called()


def test_provider_requires_evidence_source_hash_and_accepts_service_input_record(source, configuration, monkeypatch):
    image, analysis, data = source
    opener, _ = fake_transport(monkeypatch, completion(data))
    del analysis["source_sha256"]
    with pytest.raises(provider.ModelProviderError, match="analysis_source_mismatch"):
        provider.generate_plan(image, analysis)
    opener.open.assert_not_called()
    analysis["inputs"]={"fixture-asset":{"original_sha256":data["source_sha256"]}}
    assert provider.generate_plan(image, analysis).source_sha256==data["source_sha256"]


def test_missing_configuration_never_calls_network(source, monkeypatch):
    image, analysis, _ = source
    monkeypatch.delenv("VISION_LLM_API_KEY", raising=False)
    opener, _ = fake_transport(monkeypatch, b"")
    with pytest.raises(provider.ModelProviderError, match="configuration_required"):
        provider.generate_plan(image, analysis)
    opener.open.assert_not_called()


@pytest.mark.parametrize("options", [{"timeout": 61}, {"timeout": float("nan")}, {"timeout": True}, {"max_attempts": 0}, {"max_attempts": 4}, {"max_attempts": True}])
def test_transport_limits_are_validated_before_call(source, configuration, monkeypatch, options):
    image, analysis, _ = source
    opener, _ = fake_transport(monkeypatch, b"")
    with pytest.raises(provider.ModelProviderError):
        provider.generate_plan(image, analysis, **options)
    opener.open.assert_not_called()
