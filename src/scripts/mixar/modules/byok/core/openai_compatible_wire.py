# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Wire-format helpers shared by the OpenAI-compatible provider."""

from __future__ import annotations

import json
import re
from urllib.parse import urlsplit, urlunsplit

from .provider_types import ProviderError

_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def chat_messages(messages: list[dict]) -> list[dict]:
    """Strip Mixar-only continuation metadata before Chat Completions."""
    return [
        {key: val for key, val in message.items() if not key.startswith("_mixar_")}
        for message in messages
    ]


def responses_input(messages: list[dict]) -> list[dict]:
    """Translate Chat Completions history into Responses input items."""
    result = []
    for message in messages:
        continuation = message.get("_mixar_responses_output")
        if isinstance(continuation, list) and continuation:
            result.extend(item for item in continuation if isinstance(item, dict))
            continue
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
                if not isinstance(part, dict):
                    continue
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
            if not isinstance(call, dict):
                continue
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
        if not _HEADER_NAME.fullmatch(name):
            raise ValueError("Custom header name is invalid")
        text = str(val)
        if any((ord(char) < 32 and char != "\t") or ord(char) >= 127 for char in text):
            raise ValueError(f"Custom header '{name}' contains invalid characters")
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
