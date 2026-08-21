# SPDX-License-Identifier: GPL-3.0-or-later

import json
from types import SimpleNamespace

import httpx
import pytest

from mixar.modules.byok.core.agent_loop import run_agent_loop
from mixar.modules.byok.core.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    normalize_base_url,
    parse_custom_headers,
)
from mixar.modules.byok.core.provider_types import (
    ProviderCapabilities, ProviderResponse, ToolCall,
)


@pytest.mark.parametrize("raw, expected", [
    ("https://api.example.test", "https://api.example.test/v1"),
    ("https://api.example.test/", "https://api.example.test/v1"),
    ("https://api.example.test/v1/", "https://api.example.test/v1"),
])
def test_base_url_has_exactly_one_v1(raw, expected):
    assert normalize_base_url(raw) == expected


def test_custom_headers_reject_newlines():
    with pytest.raises(ValueError):
        parse_custom_headers('{"X-Test":"ok\\nBearer secret"}')


def test_list_models_and_parameter_fallback_are_conservative():
    payloads = []

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "z"}, {"id": "a"}]})
        body = json.loads(request.content)
        payloads.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: temperature"}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })

    config = OpenAICompatibleConfig(
        "https://api.example.test", "key", "model",
        temperature=0.5, streaming=False,
    )
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))
    try:
        assert provider.list_models() == ["a", "z"]
        result = provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()
    assert result.content == "ok"
    assert result.degraded_parameters == ["temperature"]
    assert "temperature" in payloads[0] and "temperature" not in payloads[1]


def test_auto_endpoint_falls_back_to_responses_only_on_missing_chat_endpoint():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return httpx.Response(200, json={
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "response ok"},
            ]}],
        })

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test", "key", "model", streaming=False,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()
    assert result.content == "response ok"
    assert paths == ["/v1/chat/completions", "/v1/responses"]


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.capabilities = ProviderCapabilities(supports_tools=True)
        self.config = SimpleNamespace(model="fake", base_url="https://fake/v1")

    def generate(self, messages, **kwargs):
        return next(self.responses)


def test_agent_runs_three_iterations_multiple_tools_and_recovers_from_error():
    provider = FakeProvider([
        ProviderResponse(tool_calls=[
            ToolCall("one", "execute_blender_python", '{"script":"bad"}'),
            ToolCall("two", "execute_blender_python", '{"script":"inspect"}'),
        ]),
        ProviderResponse(tool_calls=[
            ToolCall("three", "execute_blender_python", '{"script":"fix"}'),
        ]),
        ProviderResponse(content="Finished and verified", finish_reason="stop"),
    ])
    calls = []

    def execute(_name, args):
        calls.append(args["script"])
        if args["script"] == "bad":
            return {"success": False, "error": "boom"}
        return {"success": True, "result": args["script"]}

    result = run_agent_loop(
        provider,
        [{"role": "system", "content": "system"}, {"role": "user", "content": "build"}],
        execute,
    )

    assert result.text == "Finished and verified"
    assert calls == ["bad", "inspect", "fix"]
    assert result.debug["iterations"] == 3
    assert [item["success"] for item in result.debug["tool_results"]] == [False, True, True]
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert "boom" in tool_messages[0]["content"]
