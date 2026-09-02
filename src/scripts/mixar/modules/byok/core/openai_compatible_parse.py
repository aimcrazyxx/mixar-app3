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


def _reasoning_text(item: dict) -> str:
    parts = []
    for field in ("summary", "content"):
        value = item.get(field) or []
        if isinstance(value, list):
            for part in value:
                if isinstance(part, dict) and part.get("text"):
                    parts.append(str(part["text"]))
    return "".join(parts)


def parse_responses_body(
    body: dict,
    status_code: int,
    base_url: str,
    degraded=None,
    redact_detail=None,
) -> ProviderResponse:
    status = str(body.get("status") or "")
    if status in {"failed", "cancelled"} or body.get("error"):
        error = body.get("error") or {}
        detail = error.get("message", "") if isinstance(error, dict) else str(error)
        if redact_detail:
            detail = redact_detail(detail)
        raise user_error(status_code if status_code >= 400 else 400, detail)
    output = body.get("output")
    if not isinstance(output, list):
        raise ProviderError(
            "Provider response field 'output' must be a list",
            status_code=status_code,
        )
    text_parts = []
    reasoning_parts = []
    calls = []
    for item in output:
        if not isinstance(item, dict):
            raise ProviderError(
                "Provider response contained an invalid output item",
                status_code=status_code,
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
        if item.get("type") == "reasoning":
            reasoning_parts.append(_reasoning_text(item))
        if item.get("type") == "message":
            content = item.get("content") or []
            if not isinstance(content, list):
                raise ProviderError(
                    "Provider response message content must be a list",
                    status_code=status_code,
                )
            for part in content:
                if not isinstance(part, dict):
                    raise ProviderError(
                        "Provider response contained invalid message content",
                        status_code=status_code,
                    )
                if part.get("type") in {"output_text", "text"}:
                    text_parts.append(str(part.get("text") or ""))
    return ProviderResponse(
        content="".join(text_parts),
        tool_calls=calls,
        reasoning_content="".join(reasoning_parts),
        continuation_items=[dict(item) for item in output],
        finish_reason=status,
        endpoint=endpoint_url(base_url, "responses"),
        status_code=status_code,
        usage=body.get("usage") if isinstance(body.get("usage"), dict) else {},
        degraded_parameters=list(degraded or []),
    )


def parse_responses_response(
    response: httpx.Response, base_url: str, degraded=None, redact_detail=None
) -> ProviderResponse:
    return parse_responses_body(
        response_json(response),
        response.status_code,
        base_url,
        degraded,
        redact_detail,
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
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = ""
    return ProviderResponse(
        content=content or "",
        tool_calls=calls,
        reasoning_content=reasoning,
        finish_reason=str(choice.get("finish_reason") or ""),
        endpoint=endpoint_url(base_url, "chat/completions"),
        status_code=response.status_code,
        usage=body.get("usage") if isinstance(body.get("usage"), dict) else {},
    )


def parse_chat_stream(
    response, url: str, on_text_delta=None, redact_detail=None, cancel_event=None
) -> ProviderResponse:
    content, reasoning, finish, usage = [], [], "", {}
    calls: dict[int, dict] = {}
    saw_event = False
    for line in response.iter_lines():
        if cancel_event is not None and cancel_event.is_set():
            raise ProviderError("Agent run cancelled", category="cancelled")
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
            reasoning_delta = (
                delta.get("reasoning_content") or delta.get("reasoning") or ""
            )
            if isinstance(reasoning_delta, str) and reasoning_delta:
                reasoning.append(reasoning_delta)
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
        reasoning_content="".join(reasoning),
        finish_reason=str(finish),
        endpoint=url,
        status_code=response.status_code,
        usage=usage,
    )


def parse_responses_stream(
    response,
    url: str,
    base_url: str,
    on_text_delta=None,
    redact_detail=None,
    cancel_event=None,
) -> ProviderResponse:
    """Parse Responses API semantic SSE events without losing tool/reasoning state."""
    text_parts, reasoning_parts = [], []
    output_items: dict[int, dict] = {}
    completed = None
    saw_event = False
    for line in response.iter_lines():
        if cancel_event is not None and cancel_event.is_set():
            raise ProviderError("Agent run cancelled", category="cancelled")
        if not line or not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "Provider returned malformed Responses streaming JSON",
                status_code=response.status_code,
            ) from exc
        if not isinstance(event, dict):
            raise ProviderError(
                "Provider returned an invalid Responses streaming event",
                status_code=response.status_code,
            )
        event_type = str(event.get("type") or "")
        saw_event = saw_event or bool(event_type)
        if event_type in {"error", "response.failed"}:
            source = (
                event.get("error")
                or (event.get("response") or {}).get("error")
                or {}
            )
            detail = source.get("message", "") if isinstance(source, dict) else str(source)
            if redact_detail:
                detail = redact_detail(detail)
            raise user_error(400, detail)
        if event_type in {"response.completed", "response.incomplete"}:
            candidate = event.get("response")
            if isinstance(candidate, dict):
                completed = candidate
            continue
        if event_type == "response.output_text.delta":
            delta = event.get("delta") or ""
            if not isinstance(delta, str):
                raise ProviderError(
                    "Provider returned invalid Responses text delta",
                    status_code=response.status_code,
                )
            text_parts.append(delta)
            if delta and on_text_delta:
                on_text_delta(delta)
            continue
        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            delta = event.get("delta") or ""
            if isinstance(delta, str):
                reasoning_parts.append(delta)
            continue
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            item = event.get("item")
            if isinstance(item, dict):
                try:
                    index = int(event.get("output_index", len(output_items)))
                except (TypeError, ValueError):
                    index = len(output_items)
                output_items[index] = dict(item)
            continue
        if event_type == "response.function_call_arguments.delta":
            try:
                index = int(event.get("output_index", 0))
            except (TypeError, ValueError) as exc:
                raise ProviderError(
                    "Provider returned invalid Responses function-call index",
                    status_code=response.status_code,
                ) from exc
            item = output_items.setdefault(
                index,
                {
                    "type": "function_call",
                    "call_id": str(event.get("item_id") or ""),
                    "name": str(event.get("name") or ""),
                    "arguments": "",
                },
            )
            delta = event.get("delta") or ""
            if not isinstance(delta, str):
                raise ProviderError(
                    "Provider returned invalid Responses function arguments",
                    status_code=response.status_code,
                )
            item["arguments"] = str(item.get("arguments") or "") + delta

    if completed is not None:
        result = parse_responses_body(
            completed,
            response.status_code,
            base_url,
            redact_detail=redact_detail,
        )
        if not result.reasoning_content and reasoning_parts:
            result.reasoning_content = "".join(reasoning_parts)
        return result
    if not saw_event or not (text_parts or reasoning_parts or output_items):
        raise ProviderError(
            "Provider returned an empty Responses streaming response",
            status_code=response.status_code,
        )
    items = [item for _, item in sorted(output_items.items())]
    calls = []
    for item in items:
        if item.get("type") == "function_call":
            calls.append(
                ToolCall(
                    str(item.get("call_id") or item.get("id") or ""),
                    str(item.get("name") or ""),
                    str(item.get("arguments") or "{}"),
                )
            )
        elif item.get("type") == "reasoning":
            reasoning_parts.append(_reasoning_text(item))
    return ProviderResponse(
        content="".join(text_parts),
        tool_calls=calls,
        reasoning_content="".join(reasoning_parts),
        continuation_items=items,
        finish_reason="completed",
        endpoint=url,
        status_code=response.status_code,
    )
