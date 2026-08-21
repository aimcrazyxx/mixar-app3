# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenAI-compatible HTTP provider with conservative capability fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

from .openai_compatible_parse import (
    parse_chat_response,
    parse_chat_stream,
    parse_responses_response,
    response_json,
)
from .openai_compatible_wire import (
    endpoint_url,
    normalize_base_url,
    parse_custom_headers,
    responses_input,
    user_error,
)
from .provider_types import (
    AIProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderResponse,
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
        self.api_key = str(self.api_key or "").strip()
        self.model = str(self.model or "").strip()
        if not self.model:
            raise ValueError("Model is required")
        self.timeout = float(self.timeout)
        self.max_output_tokens = int(self.max_output_tokens)
        self.context_limit = int(self.context_limit)
        if self.timeout <= 0:
            raise ValueError("Timeout must be greater than zero")
        if self.max_output_tokens < 1:
            raise ValueError("Max output tokens must be at least 1")
        if self.context_limit < 0:
            raise ValueError("Context limit cannot be negative")
        if self.temperature is not None:
            self.temperature = float(self.temperature)
            if not 0 <= self.temperature <= 2:
                raise ValueError("Temperature must be between 0 and 2")
        if self.top_p is not None:
            self.top_p = float(self.top_p)
            if not 0 <= self.top_p <= 1:
                raise ValueError("Top P must be between 0 and 1")
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
        if not any(name.lower() == "content-type" for name in headers):
            headers["Content-Type"] = "application/json"
        if config.api_key:
            for name in list(headers):
                if name.lower() == "authorization":
                    headers.pop(name)
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
            except ValueError:
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
        return response_json(response)

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
            response = self._request(
                "POST",
                "responses",
                json={
                    "model": self.config.model,
                    "input": "Reply OK",
                    "max_output_tokens": 1,
                },
            )
            self._parse_responses_response(response)
        else:
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_tokens": 1,
            }
            try:
                try:
                    response = self._request(
                        "POST", "chat/completions", json=payload
                    )
                except ProviderError as exc:
                    if exc.status_code != 400 or not self._error_mentions(
                        exc, "max_tokens"
                    ):
                        raise
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    response = self._request(
                        "POST", "chat/completions", json=payload
                    )
                self._parse_chat_response(response)
            except ProviderError as exc:
                if self.config.endpoint_mode != "auto" or exc.status_code not in {
                    404,
                    405,
                }:
                    raise
                response = self._request(
                    "POST",
                    "responses",
                    json={
                        "model": self.config.model,
                        "input": "Reply OK",
                        "max_output_tokens": 1,
                    },
                )
                self._parse_responses_response(response)

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
                result.degraded_parameters = degraded + result.degraded_parameters
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
            "input": responses_input(messages),
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

        return self._parse_responses_response(response, degraded)

    def _parse_responses_response(
        self, response: httpx.Response, degraded=None
    ) -> ProviderResponse:
        return parse_responses_response(response, self.config.base_url, degraded)

    def _generate_json(self, payload: dict) -> ProviderResponse:
        response = self._request("POST", "chat/completions", json=payload)
        return self._parse_chat_response(response)

    def _parse_chat_response(self, response: httpx.Response) -> ProviderResponse:
        return parse_chat_response(response, self.config.base_url)

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
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    response.read()
                    result = self._parse_chat_response(response)
                    result.degraded_parameters = ["streaming"]
                    return result
                return parse_chat_stream(
                    response,
                    url,
                    on_text_delta,
                    redact_detail=self._redact_secrets,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("Connection timed out", category="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Connection failed: {type(exc).__name__}", category="connection"
            ) from exc
