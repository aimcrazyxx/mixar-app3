# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from mixar.modules.byok.core.agent_loop import run_agent_loop, trim_context
from mixar.modules.byok.core import custom_agent_runtime
from mixar.modules.byok.core.debug_report import to_json as debug_to_json
from mixar.modules.byok.core.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    normalize_base_url,
    parse_custom_headers,
)
from mixar.modules.byok.core.provider_types import (
    ProviderCapabilities,
    ProviderError,
    ProviderResponse,
    ToolCall,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://api.example.test", "https://api.example.test/v1"),
        ("https://api.example.test/", "https://api.example.test/v1"),
        ("https://api.example.test/v1/", "https://api.example.test/v1"),
        ("https://api.example.test/v1/v1", "https://api.example.test/v1"),
        (
            "https://api.example.test/openai/v1/v1/",
            "https://api.example.test/openai/v1",
        ),
    ],
)
def test_base_url_has_exactly_one_v1(raw, expected):
    assert normalize_base_url(raw) == expected


def test_base_url_rejects_embedded_credentials():
    with pytest.raises(ValueError, match="must not contain credentials"):
        normalize_base_url("https://user:secret@api.example.test")


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
            return httpx.Response(
                400, json={"error": {"message": "Unsupported parameter: temperature"}}
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    config = OpenAICompatibleConfig(
        "https://api.example.test",
        "key",
        "model",
        temperature=0.5,
        streaming=False,
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
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "response ok"},
                        ],
                    }
                ],
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test",
            "key",
            "model",
            streaming=False,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()
    assert result.content == "response ok"
    assert paths == ["/v1/chat/completions", "/v1/responses"]


def test_chat_retries_with_max_completion_tokens_when_required():
    payloads = []

    def handler(request):
        body = json.loads(request.content)
        payloads.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {"message": "Unsupported parameter: max_tokens"},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test",
            "key",
            "model",
            streaming=False,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert result.content == "ok"
    assert "max_tokens" in payloads[0]
    assert "max_completion_tokens" in payloads[1]
    assert result.degraded_parameters == ["max_tokens->max_completion_tokens"]


def test_connection_validates_selected_model_after_listing_models():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", "key", "model"),
        transport=httpx.MockTransport(handler),
    )
    try:
        provider.test_connection()
    finally:
        provider.close()

    assert paths == ["/v1/models", "/v1/chat/completions"]


def test_provider_error_redacts_non_openai_api_key():
    secret = "provider-key-12345"

    def handler(_request):
        return httpx.Response(
            401,
            json={
                "error": {"message": f"Credential {secret} is invalid"},
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", secret, "model"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ProviderError) as error:
            provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert secret not in str(error.value)
    assert "[REDACTED]" in str(error.value)


def test_chat_falls_back_to_non_streaming_when_stream_is_rejected():
    payloads = []

    def handler(request):
        body = json.loads(request.content)
        payloads.append(body)
        if body.get("stream"):
            return httpx.Response(
                400,
                json={
                    "error": {"message": "Unsupported parameter: stream"},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", "key", "model"),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert result.content == "ok"
    assert payloads[0]["stream"] is True
    assert payloads[1]["stream"] is False
    assert "stream_options" not in payloads[1]
    assert result.degraded_parameters == ["streaming"]


def test_streaming_reassembles_text_and_fragmented_tool_call():
    deltas = []
    events = [
        {"choices": [{"delta": {"content": "Working "}}]},
        {"choices": [{"delta": {"content": "now."}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "execute_blender_",
                                    "arguments": '{"script":"',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "python",
                                    "arguments": 'inspect"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 12},
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    def handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content.encode(),
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", "key", "model"),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = provider.generate(
            [{"role": "user", "content": "inspect"}],
            on_text_delta=deltas.append,
        )
    finally:
        provider.close()

    assert result.content == "Working now."
    assert deltas == ["Working ", "now."]
    assert result.finish_reason == "tool_calls"
    assert result.usage == {"total_tokens": 12}
    assert result.tool_calls == [
        ToolCall("call-1", "execute_blender_python", '{"script":"inspect"}')
    ]


def test_exact_model_auth_custom_headers_vision_and_reasoning_reach_provider():
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers.get("authorization")
        seen["tenant"] = request.headers.get("x-tenant")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test",
            "key-exact",
            "manual-model-id",
            custom_headers={"X-Tenant": "team-a"},
            reasoning_effort="high",
            streaming=False,
        ),
        transport=httpx.MockTransport(handler),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
            ],
        }
    ]
    try:
        result = provider.generate(messages)
    finally:
        provider.close()

    assert result.content == "ok"
    assert provider.capabilities.supports_vision is True
    assert provider.capabilities.supports_reasoning is True
    assert seen["authorization"] == "Bearer key-exact"
    assert seen["tenant"] == "team-a"
    assert seen["body"]["model"] == "manual-model-id"
    assert seen["body"]["reasoning_effort"] == "high"
    assert seen["body"]["messages"] == messages


def test_connection_works_when_models_endpoint_is_missing():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/models"):
            return httpx.Response(404, json={"error": {"message": "missing"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", "key", "model"),
        transport=httpx.MockTransport(handler),
    )
    try:
        provider.test_connection()
    finally:
        provider.close()

    assert paths == ["/v1/models", "/v1/chat/completions"]


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, "Invalid API key"),
        (429, "rate limit"),
        (503, "unavailable"),
    ],
)
def test_http_errors_are_classified(status, expected):
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test", "key", "model", streaming=False
        ),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                status, json={"error": {"message": "provider detail"}}
            )
        ),
    )
    try:
        with pytest.raises(ProviderError) as error:
            provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert error.value.status_code == status
    assert expected.lower() in str(error.value).lower()


def test_timeout_is_reported_without_request_details():
    def handler(_request):
        raise httpx.ReadTimeout("slow request containing sensitive body")

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", "key", "model"),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ProviderError) as error:
            provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert error.value.category == "timeout"
    assert str(error.value) == "Connection timed out"


def test_success_with_invalid_json_is_reported_as_provider_error():
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test", "key", "model", streaming=False
        ),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"not-json")
        ),
    )
    try:
        with pytest.raises(ProviderError, match="invalid JSON"):
            provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()


def test_context_trim_keeps_contiguous_recent_history_and_tool_group():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old " * 500},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "one", "function": {"name": "inspect"}}],
        },
        {"role": "tool", "tool_call_id": "one", "content": "result"},
        {"role": "user", "content": "current request"},
    ]

    trimmed = trim_context(messages, 150)

    assert [message["role"] for message in trimmed] == [
        "system",
        "assistant",
        "tool",
        "user",
    ]
    assert all("old " not in str(message.get("content")) for message in trimmed)


def test_debug_report_redacts_nested_secrets():
    report = debug_to_json(
        {
            "Authorization": "Bearer visible-secret",
            "nested": {"api_key": "secret", "message": "sk-123456789-secret"},
        }
    )

    assert "visible-secret" not in report
    assert '"api_key": "[REDACTED]"' in report
    assert "sk-123456789-secret" not in report


def test_stale_agent_finalizer_cannot_clear_newer_run_state():
    scene_name = "Scene-Race-Test"
    event = threading.Event()
    custom_agent_runtime._active_runs[scene_name] = "new-run"
    custom_agent_runtime._cancel_events[scene_name] = event
    custom_agent_runtime._stream_buffers["new-run"] = "new text"

    try:
        custom_agent_runtime._clear_run(scene_name, "old-run")
        assert custom_agent_runtime._active_runs[scene_name] == "new-run"
        assert custom_agent_runtime._cancel_events[scene_name] is event

        custom_agent_runtime._clear_run(scene_name, "new-run")
        assert scene_name not in custom_agent_runtime._active_runs
        assert scene_name not in custom_agent_runtime._cancel_events
        assert "new-run" not in custom_agent_runtime._stream_buffers
    finally:
        custom_agent_runtime._active_runs.pop(scene_name, None)
        custom_agent_runtime._cancel_events.pop(scene_name, None)
        custom_agent_runtime._stream_buffers.pop("new-run", None)


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.capabilities = ProviderCapabilities(supports_tools=True)
        self.config = SimpleNamespace(model="fake", base_url="https://fake/v1")

    def generate(self, messages, **kwargs):
        return next(self.responses)


def test_agent_cancellation_stops_before_provider_request():
    provider = FakeProvider([ProviderResponse(content="must not be consumed")])
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(ProviderError) as error:
        run_agent_loop(
            provider,
            [{"role": "user", "content": "build"}],
            lambda _name, _args: {"success": True},
            cancel_event=cancelled,
        )

    assert error.value.category == "cancelled"
    assert next(provider.responses).content == "must not be consumed"


def test_agent_runs_three_iterations_multiple_tools_and_recovers_from_error():
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall("one", "execute_blender_python", '{"script":"bad"}'),
                    ToolCall("two", "execute_blender_python", '{"script":"inspect"}'),
                ]
            ),
            ProviderResponse(
                tool_calls=[
                    ToolCall("three", "execute_blender_python", '{"script":"fix"}'),
                ]
            ),
            ProviderResponse(content="Finished and verified", finish_reason="stop"),
        ]
    )
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
    assert [item["success"] for item in result.debug["tool_results"]] == [
        False,
        True,
        True,
    ]
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert "boom" in tool_messages[0]["content"]
