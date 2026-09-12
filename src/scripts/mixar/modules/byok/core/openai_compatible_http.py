# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""HTTP retry and streaming transport for the compatible provider."""

from __future__ import annotations

import time

import httpx

from .openai_compatible_parse import parse_chat_stream, parse_responses_stream
from .openai_compatible_wire import endpoint_url, user_error
from .provider_types import ProviderError

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _retry_delay(owner, response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after", "").strip()
    try:
        return min(30.0, max(0.0, float(retry_after)))
    except ValueError:
        return min(30.0, owner.config.retry_backoff * (2**attempt))


def _wait_retry(delay: float, cancel_event) -> None:
    if cancel_event is not None:
        if cancel_event.wait(delay):
            raise ProviderError("Agent run cancelled", category="cancelled")
    elif delay:
        time.sleep(delay)


def request(owner, method: str, endpoint: str, **kwargs) -> httpx.Response:
    cancel_event = kwargs.pop("cancel_event", None)
    for attempt in range(owner.config.max_retries + 1):
        owner._cancelled(cancel_event)
        try:
            response = owner._client.request(
                method, endpoint_url(owner.config.base_url, endpoint), **kwargs
            )
        except httpx.TimeoutException as exc:
            raise ProviderError("Connection timed out", category="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Connection failed: {type(exc).__name__}", category="connection"
            ) from exc
        if (
            response.status_code not in _RETRYABLE_STATUS
            or attempt >= owner.config.max_retries
        ):
            break
        _wait_retry(_retry_delay(owner, response, attempt), cancel_event)
    if response.status_code >= 400:
        detail = ""
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else body
            detail = (
                error.get("message", "")
                if isinstance(error, dict)
                else str(error)
            )
        except ValueError:
            detail = response.text[:400]
        raise user_error(response.status_code, owner._redact_secrets(detail))
    return response


def _stream_error(owner, response: httpx.Response) -> ProviderError:
    response.read()
    detail = response.text[:400]
    try:
        data = response.json()
        detail = (data.get("error") or {}).get("message", detail)
    except ValueError:
        data = None
    return user_error(response.status_code, owner._redact_secrets(detail))


def chat_stream(owner, payload: dict, on_text_delta=None, cancel_event=None):
    url = endpoint_url(owner.config.base_url, "chat/completions")
    try:
        for attempt in range(owner.config.max_retries + 1):
            owner._cancelled(cancel_event)
            with owner._client.stream("POST", url, json=payload) as response:
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and attempt < owner.config.max_retries
                ):
                    response.read()
                    _wait_retry(_retry_delay(owner, response, attempt), cancel_event)
                    continue
                if response.status_code >= 400:
                    raise _stream_error(owner, response)
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    response.read()
                    result = owner._parse_chat_response(response)
                    result.degraded_parameters = ["streaming"]
                    owner._set_capability("supports_streaming", False)
                    return result
                return parse_chat_stream(
                    response,
                    url,
                    on_text_delta,
                    redact_detail=owner._redact_secrets,
                    cancel_event=cancel_event,
                )
    except httpx.TimeoutException as exc:
        raise ProviderError("Connection timed out", category="timeout") from exc
    except httpx.HTTPError as exc:
        raise ProviderError(
            f"Connection failed: {type(exc).__name__}", category="connection"
        ) from exc


def responses_stream(owner, payload: dict, on_text_delta=None, cancel_event=None):
    url = endpoint_url(owner.config.base_url, "responses")
    try:
        for attempt in range(owner.config.max_retries + 1):
            owner._cancelled(cancel_event)
            with owner._client.stream("POST", url, json=payload) as response:
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and attempt < owner.config.max_retries
                ):
                    response.read()
                    _wait_retry(_retry_delay(owner, response, attempt), cancel_event)
                    continue
                if response.status_code >= 400:
                    raise _stream_error(owner, response)
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    response.read()
                    result = owner._parse_responses_response(response)
                    result.degraded_parameters = ["streaming"]
                    owner._set_capability("supports_streaming", False)
                    return result
                return parse_responses_stream(
                    response,
                    url,
                    owner.config.base_url,
                    on_text_delta,
                    owner._redact_secrets,
                    cancel_event,
                )
    except httpx.TimeoutException as exc:
        raise ProviderError("Connection timed out", category="timeout") from exc
    except httpx.HTTPError as exc:
        raise ProviderError(
            f"Connection failed: {type(exc).__name__}", category="connection"
        ) from exc
