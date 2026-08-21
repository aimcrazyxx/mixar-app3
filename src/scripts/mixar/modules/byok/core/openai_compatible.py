# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenAI-compatible HTTP provider with conservative capability fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import httpx

from .openai_compatible_http import (
    chat_stream,
    responses_stream,
)
from .openai_compatible_http import (
    request as http_request,
)
from .openai_compatible_parse import (
    parse_chat_response,
    parse_responses_response,
    response_json,
)
from .openai_compatible_wire import (
    chat_messages,
    normalize_base_url,
    parse_custom_headers,
    responses_input,
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
    "stream",
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
    max_retries: int = 2
    retry_backoff: float = 0.5

    def __post_init__(self):
        self.base_url = normalize_base_url(self.base_url)
        self.api_key = str(self.api_key or "").strip()
        self.model = str(self.model or "").strip()
        if not self.model:
            raise ValueError("Model is required")
        self.timeout = float(self.timeout)
        self.max_output_tokens = int(self.max_output_tokens)
        self.context_limit = int(self.context_limit)
        self.max_retries = int(self.max_retries)
        self.retry_backoff = float(self.retry_backoff)
        if self.timeout <= 0:
            raise ValueError("Timeout must be greater than zero")
        if self.max_output_tokens < 1:
            raise ValueError("Max output tokens must be at least 1")
        if self.context_limit < 0:
            raise ValueError("Context limit cannot be negative")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("Max retries must be between 0 and 5")
        if not 0 <= self.retry_backoff <= 30:
            raise ValueError("Retry backoff must be between 0 and 30 seconds")
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
            supports_streaming=config.streaming,
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

    def _set_capability(self, name: str, value: bool) -> None:
        if getattr(self.capabilities, name) != value:
            self.capabilities = replace(self.capabilities, **{name: value})

    @staticmethod
    def _cancelled(cancel_event) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise ProviderError("Agent run cancelled", category="cancelled")

    def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        return http_request(self, method, endpoint, **kwargs)

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
        try:
            response = self._request("GET", "models")
        except ProviderError as exc:
            if exc.status_code in {404, 405}:
                self._set_capability("supports_model_listing", False)
            raise
        data = self._response_json(response)
        rows = data.get("data", [])
        if not isinstance(rows, list):
            raise ProviderError(
                "Provider model list field 'data' must be a list",
                status_code=response.status_code,
            )
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
            "messages": chat_messages(messages),
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

    def generate(
        self,
        messages,
        *,
        tools=None,
        on_text_delta=None,
        cancel_event=None,
    ) -> ProviderResponse:
        self._cancelled(cancel_event)
        if self.config.endpoint_mode == "responses":
            return self._generate_responses(
                messages, tools, on_text_delta, cancel_event
            )
        if (
            self.config.endpoint_mode == "auto"
            and not self.capabilities.supports_chat_completions
        ):
            return self._generate_responses(
                messages, tools, on_text_delta, cancel_event
            )
        stream = bool(self.config.streaming)
        payload = self._payload(messages, tools, stream)
        degraded = []
        swapped_token_parameter = False
        while True:
            try:
                if stream:
                    result = self._generate_stream(
                        payload, on_text_delta, cancel_event
                    )
                else:
                    result = self._generate_json(payload, cancel_event)
                result.degraded_parameters = degraded + result.degraded_parameters
                return result
            except ProviderError as exc:
                if self.config.endpoint_mode == "auto" and exc.status_code in {
                    404,
                    405,
                }:
                    self._set_capability("supports_chat_completions", False)
                    result = self._generate_responses(
                        messages, tools, on_text_delta, cancel_event
                    )
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
                    self._set_capability("supports_streaming", False)
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
                self._record_rejected_capability(param)
                if param == "stream_options":
                    continue
                if param == "tools":
                    payload.pop("parallel_tool_calls", None)

    def _record_rejected_capability(self, parameter: str) -> None:
        mapping = {
            "tools": "supports_tools",
            "parallel_tool_calls": "supports_parallel_tools",
            "reasoning": "supports_reasoning",
            "reasoning_effort": "supports_reasoning",
            "temperature": "supports_temperature",
            "top_p": "supports_top_p",
        }
        capability = mapping.get(parameter)
        if capability:
            self._set_capability(capability, False)

    def _generate_responses(
        self, messages, tools, on_text_delta=None, cancel_event=None
    ) -> ProviderResponse:
        payload = {
            "model": self.config.model,
            "input": responses_input(messages),
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.streaming and self.capabilities.supports_streaming:
            payload["stream"] = True
        if tools and self.capabilities.supports_tools:
            payload["tools"] = [
                {"type": "function", **item.get("function", {})}
                for item in tools
                if item.get("type") == "function"
            ]
            if self.capabilities.supports_parallel_tools:
                payload["parallel_tool_calls"] = True
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}

        degraded = []
        while True:
            try:
                if payload.get("stream"):
                    result = self._generate_responses_stream(
                        payload, on_text_delta, cancel_event
                    )
                else:
                    response = self._request(
                        "POST", "responses", json=payload, cancel_event=cancel_event
                    )
                    result = self._parse_responses_response(response)
                result.degraded_parameters = degraded + result.degraded_parameters
                return result
            except ProviderError as exc:
                if exc.status_code in {404, 405}:
                    self._set_capability("supports_responses_api", False)
                parameter = (
                    self._unsupported_parameter(exc, payload)
                    if exc.status_code == 400
                    else ""
                )
                if not parameter:
                    raise
                payload.pop(parameter, None)
                degraded.append(parameter)
                self._record_rejected_capability(parameter)
                if parameter == "tools":
                    payload.pop("parallel_tool_calls", None)
                if parameter == "stream_options":
                    continue
                if parameter == "stream":
                    self._set_capability("supports_streaming", False)

    def _parse_responses_response(
        self, response: httpx.Response, degraded=None
    ) -> ProviderResponse:
        return parse_responses_response(
            response,
            self.config.base_url,
            degraded,
            self._redact_secrets,
        )

    def _generate_json(self, payload: dict, cancel_event=None) -> ProviderResponse:
        response = self._request(
            "POST", "chat/completions", json=payload, cancel_event=cancel_event
        )
        return self._parse_chat_response(response)

    def _parse_chat_response(self, response: httpx.Response) -> ProviderResponse:
        return parse_chat_response(response, self.config.base_url)

    def _generate_stream(
        self,
        payload: dict,
        on_text_delta: Optional[Callable[[str], None]],
        cancel_event=None,
    ) -> ProviderResponse:
        return chat_stream(self, payload, on_text_delta, cancel_event)

    def _generate_responses_stream(
        self, payload: dict, on_text_delta=None, cancel_event=None
    ) -> ProviderResponse:
        return responses_stream(self, payload, on_text_delta, cancel_event)
