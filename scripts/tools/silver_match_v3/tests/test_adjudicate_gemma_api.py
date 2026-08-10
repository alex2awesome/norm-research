import json

import argparse

import pytest

from scripts.tools.silver_match_v3.adjudicate_gemma_api import (
    chat_completion,
    configure_api_access,
    reserve_api_requests,
)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()


def test_chat_completion_uses_openai_contract(monkeypatch):
    seen = {}

    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    value = chat_completion(
        base_url="http://127.0.0.1:8006/v1/",
        model="gemma",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=160,
        seed=17,
        timeout=12.0,
        transport_retries=0,
        api_key="secret-test-key",
        reasoning_effort="minimal",
        force_json_object=True,
    )
    assert value == "ok"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["seed"] == 17
    assert seen["body"]["reasoning"] == {"effort": "minimal", "exclude": True}
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["timeout"] == 12.0


def test_openrouter_requires_key_and_positive_cap(tmp_path):
    args = argparse.Namespace(
        api_base_url="https://openrouter.ai/api/v1",
        api_key_file=None,
        max_api_requests=10,
    )
    with pytest.raises(ValueError, match="api-key-file"):
        configure_api_access(args)
    key = tmp_path / "key.txt"
    key.write_text("secret\n")
    args.api_key_file = str(key)
    args.max_api_requests = 0
    with pytest.raises(ValueError, match="positive"):
        configure_api_access(args)


def test_api_request_cap_is_fail_closed(tmp_path):
    key = tmp_path / "key.txt"
    key.write_text("secret\n")
    args = argparse.Namespace(
        api_base_url="https://openrouter.ai/api/v1",
        api_key_file=str(key),
        max_api_requests=3,
    )
    configure_api_access(args)
    reserve_api_requests(args, 2)
    assert args._api_request_count == 2
    with pytest.raises(RuntimeError, match="cap exceeded"):
        reserve_api_requests(args, 2)
