# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Strict response parsers for OpenAI-compatible HTTP endpoints."""

from __future__ import annotations

import json

import httpx

from .openai_compatible_wire import endpoint_url, user_error
from .provider_types import ProviderError, ProviderResponse, ToolCall


def response_json(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderError(
            "Provider returned an invalid JSON response",
            status_code=response.status_code,
        ) from exc
    if not isinstance(body, dict):
        raise ProviderError(
            "Provider returned an unexpected JSON response",
            status_code=response.status_code,
        )
    return body


def parse_responses_response(
    response: httpx.Response, base_url: str, degraded=None
) -> ProviderResponse:
    body = response_json(response)
    output = body.get("output")
    if not isinstance(output, list):
        raise ProviderError(
            "Provider response field 'output' must be a list",
            status_code=response.status_code,
        )
    text_parts = []
    calls = []
    for item in output:
        if not isinstance(item, dict):
            raise ProviderError(
                "Provider response contained an invalid output item",
                status_code=response.status_code,
            )
        if item.get("type") == "function_call":
            arguments = item.get("arguments") or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                ToolCall(
                    str(item.get("call_id") or item.get("id") or ""),
                    str(item.get("name") or ""),
                    arguments,
                )
            )
        if item.get("type") == "message":
            content = item.get("content") or []
            if not isinstance(content, list):
                raise ProviderError(
                    "Provider response message content must be a list",
                    status_code=response.status_code,
                )
            for part in content:
                if not isinstance(part, dict):
                    raise ProviderError(
                        "Provider response contained invalid message content",
                        status_code=response.status_code,
                    )
                if part.get("type") in {"output_text", "text"}:
                    text_parts.append(str(part.get("text") or ""))
    return ProviderResponse(
        content="".join(text_parts),
        tool_calls=calls,
        finish_reason=str(body.get("status") or ""),
        endpoint=endpoint_url(base_url, "responses"),
        status_code=response.status_code,
        usage=body.get("usage") if isinstance(body.get("usage"), dict) else {},
        degraded_parameters=list(degraded or []),
    )


def parse_chat_response(response: httpx.Response, base_url: str) -> ProviderResponse:
    body = response_json(response)
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(
        choices[0], dict
    ):
        raise ProviderError(
            "Provider response did not include a valid choice",
            status_code=response.status_code,
        )
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ProviderError(
            "Provider response contained an invalid message",
            status_code=response.status_code,
        )
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ProviderError(
            "Provider response message content must be text",
            status_code=response.status_code,
        )
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list) or any(
        not isinstance(item, dict) for item in raw_calls
    ):
        raise ProviderError(
            "Provider response contained invalid tool calls",
            status_code=response.status_code,
        )
    calls = []
    for item in raw_calls:
        function = item.get("function") or {}
        if not isinstance(function, dict):
            raise ProviderError(
                "Provider response contained an invalid tool function",
                status_code=response.status_code,
            )
        arguments = function.get("arguments") or "{}"
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(
            ToolCall(
                str(item.get("id") or ""),
                str(function.get("name") or ""),
                arguments,
            )
        )
    return ProviderResponse(
        content=content or "",
        tool_calls=calls,
        finish_reason=str(choice.get("finish_reason") or ""),
        endpoint=endpoint_url(base_url, "chat/completions"),
        status_code=response.status_code,
        usage=body.get("usage") if isinstance(body.get("usage"), dict) else {},
    )


def parse_chat_stream(
    response, url: str, on_text_delta=None, redact_detail=None
) -> ProviderResponse:
    content, finish, usage = [], "", {}
    calls: dict[int, dict] = {}
    saw_event = False
    for line in response.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "Provider returned malformed streaming JSON",
                status_code=response.status_code,
            ) from exc
        if not isinstance(event, dict):
            raise ProviderError(
                "Provider returned an invalid streaming event",
                status_code=response.status_code,
            )
        if event.get("error"):
            error = event["error"]
            detail = error.get("message", "") if isinstance(error, dict) else str(error)
            if redact_detail:
                detail = redact_detail(detail)
            raise user_error(400, detail)
        if event.get("usage"):
            if not isinstance(event["usage"], dict):
                raise ProviderError(
                    "Provider returned invalid streaming usage",
                    status_code=response.status_code,
                )
            usage = event["usage"]
        choices = event.get("choices") or []
        if not isinstance(choices, list):
            raise ProviderError(
                "Provider returned invalid streaming choices",
                status_code=response.status_code,
            )
        if choices or event.get("usage"):
            saw_event = True
        for choice in choices:
            if not isinstance(choice, dict):
                raise ProviderError(
                    "Provider returned an invalid streaming choice",
                    status_code=response.status_code,
                )
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise ProviderError(
                    "Provider returned an invalid streaming delta",
                    status_code=response.status_code,
                )
            text = delta.get("content") or ""
            if text:
                if not isinstance(text, str):
                    raise ProviderError(
                        "Provider returned non-text streaming content",
                        status_code=response.status_code,
                    )
                content.append(text)
                if on_text_delta:
                    on_text_delta(text)
            tool_calls = delta.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ProviderError(
                    "Provider returned invalid streaming tool calls",
                    status_code=response.status_code,
                )
            for item in tool_calls:
                if not isinstance(item, dict):
                    raise ProviderError(
                        "Provider returned an invalid streaming tool call",
                        status_code=response.status_code,
                    )
                try:
                    idx = int(item.get("index", 0))
                except (TypeError, ValueError) as exc:
                    raise ProviderError(
                        "Provider returned an invalid tool-call index",
                        status_code=response.status_code,
                    ) from exc
                slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                slot["id"] += str(item.get("id") or "")
                function = item.get("function") or {}
                if not isinstance(function, dict):
                    raise ProviderError(
                        "Provider returned an invalid streaming tool function",
                        status_code=response.status_code,
                    )
                slot["name"] += str(function.get("name") or "")
                arguments = function.get("arguments") or ""
                if not isinstance(arguments, str):
                    raise ProviderError(
                        "Provider returned invalid streaming tool arguments",
                        status_code=response.status_code,
                    )
                slot["arguments"] += arguments
    if not saw_event:
        raise ProviderError(
            "Provider returned an empty streaming response",
            status_code=response.status_code,
        )
    return ProviderResponse(
        content="".join(content),
        tool_calls=[
            ToolCall(value["id"], value["name"], value["arguments"])
            for _, value in sorted(calls.items())
        ],
        finish_reason=str(finish),
        endpoint=url,
        status_code=response.status_code,
        usage=usage,
    )
