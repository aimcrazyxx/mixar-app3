# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from mixar.modules.byok.core import custom_agent_runtime
from mixar.modules.byok.core.agent_loop import run_agent_loop, trim_context
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


@pytest.mark.parametrize(
    "headers",
    [
        {"Bad Header": "value"},
        {"X-Test": "value\x00"},
        {"X-Test": "não-ascii"},
    ],
)
def test_custom_headers_reject_invalid_http_characters(headers):
    with pytest.raises(ValueError):
        parse_custom_headers(headers)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"timeout": 0}, "Timeout"),
        ({"max_output_tokens": 0}, "Max output"),
        ({"context_limit": -1}, "Context"),
        ({"temperature": 2.1}, "Temperature"),
        ({"top_p": 1.1}, "Top P"),
    ],
)
def test_config_rejects_invalid_numeric_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleConfig(
            "https://api.example.test", "key", "model", **kwargs
        )


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
    assert provider.capabilities.supports_temperature is False


def test_model_listing_rejects_malformed_success_payload():
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", "key", "model"),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"data": {"id": "wrong"}})
        ),
    )
    try:
        with pytest.raises(ProviderError, match="must be a list"):
            provider.list_models()
    finally:
        provider.close()


def test_transient_http_errors_are_retried_with_a_bound():
    attempts = []

    def handler(_request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, json={"error": {"message": "temporary"}})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test",
            "key",
            "model",
            streaming=False,
            retry_backoff=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert result.content == "ok"
    assert len(attempts) == 3


def test_retry_backoff_is_cancellable():
    cancelled = threading.Event()
    timer = threading.Timer(0.01, cancelled.set)
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test",
            "key",
            "model",
            streaming=False,
            retry_backoff=30,
        ),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(429, json={"error": {"message": "busy"}})
        ),
    )
    timer.start()
    try:
        with pytest.raises(ProviderError) as error:
            provider.generate(
                [{"role": "user", "content": "hello"}],
                cancel_event=cancelled,
            )
    finally:
        timer.cancel()
        provider.close()

    assert error.value.category == "cancelled"


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


def test_stream_request_accepts_json_response_from_compatible_provider():
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", "key", "model"),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "choices": [
                        {"message": {"content": "json fallback"}, "finish_reason": "stop"}
                    ]
                },
            )
        ),
    )
    try:
        result = provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert result.content == "json fallback"
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


def test_chat_reasoning_is_returned_to_model_after_tool_execution():
    payloads = []

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": "inspect first",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "execute_blender_python",
                                            "arguments": '{"script":"inspect"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "verified"}, "finish_reason": "stop"}
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test", "key", "model", streaming=False
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = run_agent_loop(
            provider,
            [{"role": "user", "content": "inspect"}],
            lambda _name, _args: {"success": True, "result": "ok"},
        )
    finally:
        provider.close()

    assistant = next(
        message for message in payloads[1]["messages"] if message["role"] == "assistant"
    )
    assert assistant["reasoning_content"] == "inspect first"
    assert result.text == "verified"
    assert result.debug["reasoning_present"] is True


def test_responses_stream_preserves_reasoning_tool_state_usage_and_deltas():
    payloads = []
    deltas = []
    reasoning_item = {
        "id": "rs-1",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "inspect first"}],
    }
    call_item = {
        "id": "fc-1",
        "type": "function_call",
        "call_id": "call-1",
        "name": "execute_blender_python",
        "arguments": '{"script":"inspect"}',
    }

    def sse(events):
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(content + "data: [DONE]\n\n").encode(),
        )

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if len(payloads) == 1:
            return sse(
                [
                    {"type": "response.output_text.delta", "delta": "Inspecting. "},
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "output": [reasoning_item, call_item],
                            "usage": {"total_tokens": 10},
                        },
                    },
                ]
            )
        return sse(
            [
                {"type": "response.output_text.delta", "delta": "Done."},
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "Done."}
                                ],
                            }
                        ],
                        "usage": {"total_tokens": 7},
                    },
                },
            ]
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test",
            "key",
            "model",
            endpoint_mode="responses",
            reasoning_effort="high",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = run_agent_loop(
            provider,
            [{"role": "user", "content": "inspect"}],
            lambda _name, _args: {"success": True, "result": "ok"},
            on_text_delta=deltas.append,
        )
    finally:
        provider.close()

    assert payloads[0]["stream"] is True
    assert reasoning_item in payloads[1]["input"]
    assert call_item in payloads[1]["input"]
    assert {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": '{"success": true, "result": "ok"}',
    } in payloads[1]["input"]
    assert deltas == ["Inspecting. ", "Done."]
    assert result.text == "Done."
    assert result.debug["request_count"] == 2
    assert result.debug["reasoning_present"] is True
    assert [item["values"]["total_tokens"] for item in result.debug["usage"]] == [
        10,
        7,
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


def test_api_key_overrides_case_variant_custom_authorization_header():
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
            },
        )

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test",
            "  exact-key  ",
            "model",
            custom_headers={"authorization": "Bearer stale-key"},
            streaming=False,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert seen["authorization"] == "Bearer exact-key"


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


@pytest.mark.parametrize("endpoint_mode", ["auto", "responses"])
def test_connection_rejects_unrecognized_success_payload(endpoint_mode):
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(404, json={"error": {"message": "missing"}})
        return httpx.Response(200, json={"unexpected": True})

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test",
            "key",
            "model",
            endpoint_mode=endpoint_mode,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ProviderError, match=r"choice|output"):
            provider.test_connection()
    finally:
        provider.close()


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


def test_malformed_chat_tool_function_is_reported_as_provider_error():
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            "https://api.example.test", "key", "model", streaming=False
        ),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [{"id": "one", "function": "bad"}],
                            }
                        }
                    ]
                },
            )
        ),
    )
    try:
        with pytest.raises(ProviderError, match="tool function"):
            provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()


@pytest.mark.parametrize(
    "content, message",
    [
        (b"data: [DONE]\n\n", "empty streaming"),
        (b"data: {}\n\ndata: [DONE]\n\n", "empty streaming"),
        (b"data: not-json\n\n", "malformed streaming"),
    ],
)
def test_invalid_stream_is_not_accepted_as_an_empty_success(content, message):
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", "key", "model"),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
            )
        ),
    )
    try:
        with pytest.raises(ProviderError, match=message):
            provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()


def test_stream_error_redacts_provider_secret():
    secret = "provider-secret-123"
    event = {"error": {"message": f"Credential {secret} is invalid"}}
    content = f"data: {json.dumps(event)}\n\n".encode()
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig("https://api.example.test", secret, "model"),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
            )
        ),
    )
    try:
        with pytest.raises(ProviderError) as error:
            provider.generate([{"role": "user", "content": "hello"}])
    finally:
        provider.close()

    assert secret not in str(error.value)
    assert "[REDACTED]" in str(error.value)


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


def test_context_trim_never_drops_current_request_after_tool_calls():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "previous " * 500},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current build request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "one", "function": {"name": "inspect"}}],
        },
        {"role": "tool", "tool_call_id": "one", "content": "scene state"},
    ]

    trimmed = trim_context(messages, 80)

    assert [message["role"] for message in trimmed] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert trimmed[1]["content"] == "current build request"
    assert "previous" not in str(trimmed)


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


def test_debug_report_keeps_non_secret_token_counts():
    report = debug_to_json({"context_tokens_approx": 321, "access_token": "secret"})

    assert '"context_tokens_approx": 321' in report
    assert '"access_token": "[REDACTED]"' in report


@pytest.mark.parametrize(
    "enabled, provider, route, expected",
    [
        (True, "openai-compatible", "DIRECT", True),
        (True, "openai-compatible", "MIXAR", False),
        (True, "openai-compatible", "", False),
        (True, "openai", "DIRECT", False),
        (True, "openrouter", "DIRECT", False),
        (True, " openai-compatible", "DIRECT", False),
        (True, "OPENAI-COMPATIBLE", "DIRECT", False),
        (False, "openai-compatible", "DIRECT", False),
        (True, "", "DIRECT", False),
    ],
)
def test_direct_route_requires_explicit_custom_provider_selection(
    enabled, provider, route, expected
):
    wm = SimpleNamespace(
        byok_custom_enabled=enabled,
        byok_current_provider=provider,
        byok_custom_active_route=route,
    )

    assert custom_agent_runtime.is_active(wm) is expected


def test_unsaved_route_edit_does_not_change_active_chat_route():
    wm = SimpleNamespace(
        byok_custom_enabled=True,
        byok_current_provider="openai-compatible",
        byok_custom_active_route="DIRECT",
        byok_custom_route="MIXAR",
    )

    assert custom_agent_runtime.is_active(wm) is True


def test_switching_from_backend_to_direct_starts_a_fresh_wire_session():
    scene = SimpleNamespace(name="Route-Test", mixie_session_id="backend-session")

    changed = custom_agent_runtime.prepare_session_route(scene, True)

    assert changed is True
    assert scene.mixie_session_id == ""


def test_switching_from_direct_to_backend_never_reuses_local_session_id():
    session_id = "direct-session"
    scene = SimpleNamespace(name="Route-Test", mixie_session_id=session_id)
    custom_agent_runtime._remember_custom_session(session_id)
    custom_agent_runtime._session_histories[session_id] = [
        {"role": "user", "content": "local only"}
    ]

    try:
        assert custom_agent_runtime.prepare_session_route(scene, True) is False
        assert scene.mixie_session_id == session_id

        assert custom_agent_runtime.prepare_session_route(scene, False) is True
        assert scene.mixie_session_id == ""
        assert custom_agent_runtime.is_custom_session(session_id) is False
        assert session_id not in custom_agent_runtime._session_histories
    finally:
        custom_agent_runtime.forget_session(session_id)


def test_backend_route_preserves_unknown_backend_owned_session_id():
    scene = SimpleNamespace(name="Route-Test", mixie_session_id="backend-session")

    assert custom_agent_runtime.prepare_session_route(scene, False) is False
    assert scene.mixie_session_id == "backend-session"


def test_prefixed_direct_session_remains_identifiable_after_runtime_reload():
    session_id = custom_agent_runtime.CUSTOM_SESSION_PREFIX + "saved-session"
    scene = SimpleNamespace(name="Route-Test", mixie_session_id=session_id)

    assert custom_agent_runtime.is_custom_session(session_id) is True
    assert custom_agent_runtime.prepare_session_route(scene, False) is True
    assert scene.mixie_session_id == ""


def test_direct_multimodal_message_keeps_backend_image_content_shape():
    message = custom_agent_runtime._with_images(
        {"role": "user", "content": "project rules\n\nbuild this"},
        [{"mime_type": "image/webp", "base64": "d2VicA=="}],
    )

    assert message == {
        "role": "user",
        "content": [
            {"type": "text", "text": "project rules\n\nbuild this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/webp;base64,d2VicA=="},
            },
        ],
    }


def test_compatible_provider_modules_respect_project_line_limit():
    core = Path(__file__).parents[1] / "src/scripts/mixar/modules/byok/core"

    for filename in (
        "openai_compatible.py",
        "openai_compatible_http.py",
        "openai_compatible_parse.py",
        "openai_compatible_wire.py",
        "agent_loop.py",
        "custom_agent_runtime.py",
    ):
        assert len((core / filename).read_text(encoding="utf-8").splitlines()) <= 500


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


def test_queued_finalizer_observes_new_chat_cancellation():
    scene_name = "Scene-Finalizer-Cancel-Test"
    event = threading.Event()
    custom_agent_runtime._cancel_events[scene_name] = event

    try:
        assert custom_agent_runtime._was_cancelled(scene_name) is False
        event.set()
        assert custom_agent_runtime._was_cancelled(scene_name) is True
    finally:
        custom_agent_runtime._cancel_events.pop(scene_name, None)


def test_timed_out_queued_blender_tool_is_not_executed_later(monkeypatch):
    scheduled = []
    executions = []
    executor_module = ModuleType("mixar.modules.space_mixie_chat.core.executor")
    executor_module.get_executor = lambda: SimpleNamespace(
        execute=lambda script: executions.append(script)
    )
    main_thread_module = ModuleType(
        "mixar.modules.space_mixie_chat.core.main_thread_executor"
    )
    main_thread_module.run_on_main_thread = scheduled.append
    monkeypatch.setitem(sys.modules, executor_module.__name__, executor_module)
    monkeypatch.setitem(sys.modules, main_thread_module.__name__, main_thread_module)

    result = custom_agent_runtime._execute_tool_sync(
        "must_not_run", 0.01, threading.Event()
    )
    scheduled[0]()

    assert result == {
        "success": False,
        "error": "Blender tool execution timed out",
    }
    assert executions == []


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


def test_agent_synthesizes_missing_tool_call_id():
    provider = FakeProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ToolCall("", "execute_blender_python", '{"script":"inspect"}')
                ]
            ),
            ProviderResponse(content="done", finish_reason="stop"),
        ]
    )

    result = run_agent_loop(
        provider,
        [{"role": "user", "content": "inspect"}],
        lambda _name, _args: {"success": True},
    )

    assistant = next(message for message in result.messages if message.get("tool_calls"))
    tool = next(message for message in result.messages if message.get("role") == "tool")
    assert assistant["tool_calls"][0]["id"] == "call_1_0"
    assert tool["tool_call_id"] == "call_1_0"


def test_important_capability_degradation_is_visible_to_user():
    provider = FakeProvider(
        [
            ProviderResponse(
                content="I can only explain the steps.",
                finish_reason="stop",
                degraded_parameters=["tools", "reasoning_effort"],
            )
        ]
    )

    result = run_agent_loop(
        provider,
        [{"role": "user", "content": "build the scene"}],
        lambda _name, _args: {"success": True},
    )

    assert "tool calling" in result.text
    assert "reasoning effort" in result.text
    assert len(result.debug["capability_warnings"]) == 2
