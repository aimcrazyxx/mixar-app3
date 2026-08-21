# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenAI-compatible HTTP provider with conservative capability fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from .provider_types import (
    AIProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderResponse,
    ToolCall,
)

_OPTIONAL_PARAMS = (
    "parallel_tool_calls",
    "reasoning_effort",
    "stream_options",
    "reasoning",
    "temperature",
    "top_p",
    "tools",
    "max_output_tokens",
)


def _responses_input(messages: list[dict]) -> list[dict]:
    """Translate Chat Completions history into Responses input items."""
    result = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        content = message.get("content") or ""
        if isinstance(content, list):
            converted = []
            for part in content:
                if part.get("type") == "text":
                    converted.append(
                        {"type": "input_text", "text": part.get("text", "")}
                    )
                elif part.get("type") == "image_url":
                    image = part.get("image_url") or {}
                    converted.append(
                        {"type": "input_image", "image_url": image.get("url", "")}
                    )
            content = converted
        if content:
            result.append({"role": role, "content": content})
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            result.append(
                {
                    "type": "function_call",
                    "call_id": str(call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or "{}"),
                }
            )
    return result


def normalize_base_url(value: str) -> str:
    """Return one canonical API root ending in exactly one ``/v1``."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Base URL is required")
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("Base URL must be an http:// or https:// URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Base URL must not contain credentials")
    path = re.sub(r"/+", "/", parts.path or "").rstrip("/")
    path = re.sub(r"(?:/v1)+$", "", path)
    path += "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def endpoint_url(base_url: str, endpoint: str) -> str:
    return f"{normalize_base_url(base_url)}/{str(endpoint).lstrip('/')}"


def parse_custom_headers(value) -> dict[str, str]:
    if value in (None, "", {}):
        return {}
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        raise ValueError("Custom headers must be a JSON object")
    result = {}
    for key, val in data.items():
        name = str(key).strip()
        if not name or "\n" in name or "\r" in name:
            raise ValueError("Custom header name is invalid")
        text = str(val)
        if "\n" in text or "\r" in text:
            raise ValueError(f"Custom header '{name}' is invalid")
        result[name] = text
    return result


def user_error(status: int, detail: str = "") -> ProviderError:
    safe_detail = str(detail or "").replace("\r", " ").replace("\n", " ")[:400]
    safe_detail = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", safe_detail)
    safe_detail = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "sk-[REDACTED]", safe_detail)
    labels = {
        400: "Provider rejected the request",
        401: "Invalid API key",
        403: "Access denied by provider",
        404: "Endpoint is not supported",
        408: "Provider request timed out",
        429: "Provider rate limit reached",
    }
    base = labels.get(
        status,
        "Provider is unavailable" if status >= 500 else "Provider request failed",
    )
    return ProviderError(
        f"{base}{': ' + safe_detail if safe_detail else ''}", status_code=status
    )


@dataclass
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0
    max_output_tokens: int = 8192
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: str = ""
    custom_headers: dict[str, str] = field(default_factory=dict)
    context_limit: int = 0
    tool_calling: bool = True
    vision: bool = True
    streaming: bool = True
    endpoint_mode: str = "auto"

    def __post_init__(self):
        self.base_url = normalize_base_url(self.base_url)
        self.model = str(self.model or "").strip()
        if not self.model:
            raise ValueError("Model is required")
        self.custom_headers = parse_custom_headers(self.custom_headers)
        if self.endpoint_mode not in {"auto", "chat_completions", "responses"}:
            raise ValueError("Invalid endpoint mode")


class OpenAICompatibleProvider(AIProvider):
    def __init__(self, config: OpenAICompatibleConfig, *, transport=None):
        self.config = config
        self.capabilities = ProviderCapabilities(
            supports_tools=config.tool_calling,
            supports_vision=config.vision,
            supports_streaming=(
                config.streaming and config.endpoint_mode != "responses"
            ),
            supports_reasoning=bool(config.reasoning_effort),
            supports_responses_api=True,
        )
        headers = dict(config.custom_headers)
        headers.setdefault("Content-Type", "application/json")
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        timeout = httpx.Timeout(config.timeout, connect=min(config.timeout, 15.0))
        self._client = httpx.Client(
            headers=headers, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(
                method, endpoint_url(self.config.base_url, endpoint), **kwargs
            )
        except httpx.TimeoutException as exc:
            raise ProviderError("Connection timed out", category="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Connection failed: {type(exc).__name__}", category="connection"
            ) from exc
        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                err = body.get("error", body) if isinstance(body, dict) else body
                detail = err.get("message", "") if isinstance(err, dict) else str(err)
            except Exception:
                detail = response.text[:400]
            raise user_error(response.status_code, self._redact_secrets(detail))
        return response

    def _redact_secrets(self, detail: str) -> str:
        safe = str(detail or "")
        candidates = [self.config.api_key, *self.config.custom_headers.values()]
        for value in candidates:
            secret = str(value or "")
            if len(secret) >= 4:
                safe = safe.replace(secret, "[REDACTED]")
        return safe

    @staticmethod
    def _response_json(response: httpx.Response) -> dict:
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

    def list_models(self) -> list[str]:
        response = self._request("GET", "models")
        data = self._response_json(response)
        rows = data.get("data", [])
        return sorted(
            {
                str(row.get("id"))
                for row in rows
                if isinstance(row, dict) and row.get("id")
            }
        )

    def test_connection(self) -> None:
        try:
            self.list_models()
        except ProviderError as exc:
            if exc.status_code not in {404, 405}:
                raise
        if self.config.endpoint_mode == "responses":
            self._request(
                "POST",
                "responses",
                json={
                    "model": self.config.model,
                    "input": "Reply OK",
                    "max_output_tokens": 1,
                },
            )
        else:
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_tokens": 1,
            }
            try:
                try:
                    self._request("POST", "chat/completions", json=payload)
                except ProviderError as exc:
                    if exc.status_code != 400 or not self._error_mentions(
                        exc, "max_tokens"
                    ):
                        raise
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    self._request("POST", "chat/completions", json=payload)
            except ProviderError as exc:
                if self.config.endpoint_mode != "auto" or exc.status_code not in {
                    404,
                    405,
                }:
                    raise
                self._request(
                    "POST",
                    "responses",
                    json={
                        "model": self.config.model,
                        "input": "Reply OK",
                        "max_output_tokens": 1,
                    },
                )

    def _payload(
        self, messages: list[dict], tools: Optional[list[dict]], stream: bool
    ) -> dict:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_output_tokens,
            "stream": stream,
        }
        if tools and self.capabilities.supports_tools:
            payload["tools"] = tools
            if self.capabilities.supports_parallel_tools:
                payload["parallel_tool_calls"] = True
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _unsupported_parameter(error: ProviderError, payload: dict) -> str:
        text = str(error).lower()
        for name in _OPTIONAL_PARAMS:
            if name in payload and re.search(rf"\b{re.escape(name.lower())}\b", text):
                return name
        return ""

    @staticmethod
    def _error_mentions(error: ProviderError, parameter: str) -> bool:
        return bool(
            re.search(
                rf"\b{re.escape(parameter.lower())}\b",
                str(error).lower(),
            )
        )

    def generate(self, messages, *, tools=None, on_text_delta=None) -> ProviderResponse:
        if self.config.endpoint_mode == "responses":
            return self._generate_responses(messages, tools)
        stream = bool(self.config.streaming)
        payload = self._payload(messages, tools, stream)
        degraded = []
        swapped_token_parameter = False
        while True:
            try:
                if stream:
                    result = self._generate_stream(payload, on_text_delta)
                else:
                    result = self._generate_json(payload)
                result.degraded_parameters = degraded
                return result
            except ProviderError as exc:
                if self.config.endpoint_mode == "auto" and exc.status_code in {
                    404,
                    405,
                }:
                    result = self._generate_responses(messages, tools)
                    result.degraded_parameters = degraded + result.degraded_parameters
                    return result
                if (
                    exc.status_code == 400
                    and "max_tokens" in payload
                    and not swapped_token_parameter
                    and self._error_mentions(exc, "max_tokens")
                ):
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    degraded.append("max_tokens->max_completion_tokens")
                    swapped_token_parameter = True
                    continue
                if (
                    exc.status_code == 400
                    and stream
                    and self._error_mentions(exc, "stream")
                ):
                    stream = False
                    payload["stream"] = False
                    payload.pop("stream_options", None)
                    degraded.append("streaming")
                    continue
                param = (
                    self._unsupported_parameter(exc, payload)
                    if exc.status_code == 400
                    else ""
                )
                if not param:
                    raise
                payload.pop(param, None)
                degraded.append(param)
                if param == "stream_options":
                    continue
                if param == "tools":
                    payload.pop("parallel_tool_calls", None)

    def _generate_responses(self, messages, tools) -> ProviderResponse:
        payload = {
            "model": self.config.model,
            "input": _responses_input(messages),
            "max_output_tokens": self.config.max_output_tokens,
        }
        if tools and self.capabilities.supports_tools:
            payload["tools"] = [
                {"type": "function", **item.get("function", {})}
                for item in tools
                if item.get("type") == "function"
            ]
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}

        degraded = []
        while True:
            try:
                response = self._request("POST", "responses", json=payload)
                break
            except ProviderError as exc:
                parameter = (
                    self._unsupported_parameter(exc, payload)
                    if exc.status_code == 400
                    else ""
                )
                if not parameter:
                    raise
                payload.pop(parameter, None)
                degraded.append(parameter)

        body = self._response_json(response)
        text_parts = []
        calls = []
        for item in body.get("output") or []:
            if item.get("type") == "function_call":
                calls.append(
                    ToolCall(
                        str(item.get("call_id") or item.get("id") or ""),
                        str(item.get("name") or ""),
                        str(item.get("arguments") or "{}"),
                    )
                )
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") in {"output_text", "text"}:
                        text_parts.append(str(part.get("text") or ""))
        return ProviderResponse(
            content="".join(text_parts),
            tool_calls=calls,
            finish_reason=str(body.get("status") or ""),
            endpoint=endpoint_url(self.config.base_url, "responses"),
            status_code=response.status_code,
            usage=body.get("usage") or {},
            degraded_parameters=degraded,
        )

    def _generate_json(self, payload: dict) -> ProviderResponse:
        response = self._request("POST", "chat/completions", json=payload)
        body = self._response_json(response)
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = [
            ToolCall(
                str(item.get("id") or ""),
                str((item.get("function") or {}).get("name") or ""),
                str((item.get("function") or {}).get("arguments") or "{}"),
            )
            for item in (message.get("tool_calls") or [])
        ]
        return ProviderResponse(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            finish_reason=str(choice.get("finish_reason") or ""),
            endpoint=endpoint_url(self.config.base_url, "chat/completions"),
            status_code=response.status_code,
            usage=body.get("usage") or {},
        )

    def _generate_stream(
        self, payload: dict, on_text_delta: Optional[Callable[[str], None]]
    ) -> ProviderResponse:
        url = endpoint_url(self.config.base_url, "chat/completions")
        try:
            with self._client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    detail = response.text[:400]
                    try:
                        data = response.json()
                        detail = (data.get("error") or {}).get("message", detail)
                    except ValueError:
                        # Non-JSON provider errors are valid; keep the bounded
                        # response text captured above instead of hiding it.
                        data = None
                    raise user_error(
                        response.status_code,
                        self._redact_secrets(detail),
                    )
                content, finish, usage = [], "", {}
                calls: dict[int, dict] = {}
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        if text:
                            content.append(text)
                            if on_text_delta:
                                on_text_delta(text)
                        for item in delta.get("tool_calls") or []:
                            idx = int(item.get("index", 0))
                            slot = calls.setdefault(
                                idx, {"id": "", "name": "", "arguments": ""}
                            )
                            slot["id"] += str(item.get("id") or "")
                            fn = item.get("function") or {}
                            slot["name"] += str(fn.get("name") or "")
                            slot["arguments"] += str(fn.get("arguments") or "")
                return ProviderResponse(
                    content="".join(content),
                    tool_calls=[
                        ToolCall(v["id"], v["name"], v["arguments"])
                        for _, v in sorted(calls.items())
                    ],
                    finish_reason=str(finish),
                    endpoint=url,
                    status_code=response.status_code,
                    usage=usage,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("Connection timed out", category="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Connection failed: {type(exc).__name__}", category="connection"
            ) from exc
