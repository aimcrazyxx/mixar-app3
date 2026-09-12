# SPDX-FileCopyrightText: 2026 Mixar Authors
# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Secure device relay for Mixar-orchestrated OpenAI-compatible requests.

The Mixar backend owns the full agent graph and Blender toolset, but it never
receives the user's provider credential. Instead it sends an ``llm.request``
JSON-RPC request to Blender. This module accepts only the exact API endpoint
approved in the locally stored provider configuration and injects locally
stored authentication immediately before the HTTP request.

The backend cannot change the scheme, host, port, path, method, credentials,
or custom headers. Redirects and oversized bodies are rejected.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Optional
from urllib.parse import SplitResult, urlsplit

from mixar.config.logging_config import get_logger
from mixar.modules.common.secure_storage import get_secret

from ..constants import OPENAI_COMPATIBLE_ROUTE_MIXAR
from .openai_compatible_wire import normalize_base_url, parse_custom_headers

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 120.0
_MAX_TIMEOUT = 280.0
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_CONFIG_SECRET = "openai_compatible_config"
_API_KEY_SECRET = "openai_compatible_api_key"

_BACKEND_HEADER_ALLOWLIST = frozenset({"accept", "content-type"})
_LOCAL_HEADER_DENYLIST = frozenset(
    {
        "accept-encoding",
        "connection",
        "content-length",
        "host",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {"content-type", "retry-after", "x-request-id"}
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect())


def _error_response(status_code: int, message: str) -> dict[str, Any]:
    body = json.dumps(
        {"error": {"message": message, "type": "mixar_openai_relay"}}
    )
    return {
        "status_code": int(status_code),
        "headers": {"content-type": "application/json"},
        "body": body,
    }


def _approved_config() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Return the locally approved relay config, failing closed."""
    raw = get_secret(_CONFIG_SECRET)
    if not raw:
        return None, "OpenAI-compatible relay is not configured on this device."
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("configuration must be an object")
        if data.get("route") != OPENAI_COMPATIBLE_ROUTE_MIXAR:
            return None, "The OpenAI-compatible provider is not using the Mixar route."
        data["base_url"] = normalize_base_url(data.get("base_url", ""))
        data["custom_headers"] = parse_custom_headers(
            data.get("custom_headers") or {}
        )
        return data, None
    except Exception as exc:
        logger.warning(
            "Could not load OpenAI-compatible relay approval: %s",
            type(exc).__name__,
        )
        return None, "The locally stored OpenAI-compatible approval is invalid."


def get_approved_base_url() -> str:
    """Expose the local trust anchor for diagnostics and tests."""
    config, _ = _approved_config()
    return str(config.get("base_url") or "") if config else ""


def _effective_port(parts: SplitResult) -> int:
    return parts.port or (443 if parts.scheme.lower() == "https" else 80)


def _target_matches_approval(method: str, url: str, base_url: str) -> bool:
    """Allow only generation endpoints derived from the approved API root."""
    try:
        approved = urlsplit(normalize_base_url(base_url))
        target = urlsplit(str(url or ""))
        target_port = _effective_port(target)
        approved_port = _effective_port(approved)
    except (TypeError, ValueError):
        return False
    if method != "POST" or target.scheme.lower() not in {"http", "https"}:
        return False
    if not target.hostname or target.username or target.password:
        return False
    if target.query or target.fragment:
        return False
    root_path = approved.path.rstrip("/")
    allowed_paths = {
        f"{root_path}/chat/completions",
        f"{root_path}/responses",
    }
    return (
        target.scheme.lower() == approved.scheme.lower()
        and target.hostname.lower() == (approved.hostname or "").lower()
        and target_port == approved_port
        and target.path in allowed_paths
    )


def _coerce_timeout(value: Any, configured: Any) -> float:
    try:
        ceiling = min(_MAX_TIMEOUT, max(1.0, float(configured)))
    except (TypeError, ValueError):
        ceiling = _DEFAULT_TIMEOUT
    try:
        return min(ceiling, max(1.0, float(value)))
    except (TypeError, ValueError):
        return ceiling


def _request_headers(params: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    """Build headers from local configuration, never backend credentials."""
    result: dict[str, str] = {}
    backend_headers = params.get("headers") or {}
    if isinstance(backend_headers, dict):
        for name, value in backend_headers.items():
            lowered = str(name).lower()
            if lowered in _BACKEND_HEADER_ALLOWLIST:
                result[str(name)] = str(value)

    # These headers were typed by the user and stored in the OS credential
    # store. Unlike backend headers, they are part of the local trust anchor.
    for name, value in (config.get("custom_headers") or {}).items():
        if name.lower() not in _LOCAL_HEADER_DENYLIST:
            result[name] = value

    if not any(name.lower() == "content-type" for name in result):
        result["Content-Type"] = "application/json"

    api_key = get_secret(_API_KEY_SECRET).strip()
    if api_key:
        for name in list(result):
            if name.lower() == "authorization":
                result.pop(name)
        result["Authorization"] = f"Bearer {api_key}"
    return result


def _read_response_body(stream) -> Optional[str]:
    raw = stream.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        return None
    return raw.decode("utf-8", errors="replace")


def _redact_local_secrets(value: str, config: dict[str, Any]) -> str:
    """Prevent a provider response from echoing local auth to the backend."""
    safe = str(value or "")
    candidates = [
        get_secret(_API_KEY_SECRET),
        *(config.get("custom_headers") or {}).values(),
    ]
    for candidate in candidates:
        secret = str(candidate or "")
        if len(secret) >= 4:
            safe = safe.replace(secret, "[REDACTED]")
    return safe


def _response_headers(headers, config: dict[str, Any]) -> dict[str, str]:
    return {
        str(name): _redact_local_secrets(str(value), config)
        for name, value in (headers or {}).items()
        if str(name).lower() in _RESPONSE_HEADER_ALLOWLIST
    }


def handle_llm_request(params: dict[str, Any]) -> dict[str, Any]:
    """Execute one approved relay request and always return an HTTP envelope."""
    if not isinstance(params, dict):
        return _error_response(400, "Relay parameters must be an object.")

    config, approval_error = _approved_config()
    if not config:
        return _error_response(403, approval_error or "Relay is not approved.")

    method = str(params.get("method") or "POST").upper()
    url = str(params.get("url") or "")
    if not _target_matches_approval(method, url, config["base_url"]):
        logger.warning("Refusing OpenAI-compatible relay target: %s %s", method, url)
        return _error_response(
            403,
            "The relayed request does not match the endpoint approved on this device.",
        )

    body = params.get("body") or ""
    if isinstance(body, str):
        data = body.encode("utf-8")
    elif isinstance(body, (bytes, bytearray)):
        data = bytes(body)
    else:
        return _error_response(400, "Relay request body must be text or bytes.")
    if len(data) > _MAX_REQUEST_BYTES:
        return _error_response(413, "Relay request exceeded the 16 MiB limit.")

    request = urllib.request.Request(url, data=data, method=method)
    for name, value in _request_headers(params, config).items():
        request.add_header(name, value)
    timeout = _coerce_timeout(params.get("timeout"), config.get("timeout"))

    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
            response_body = _read_response_body(response)
            if response_body is None:
                return _error_response(502, "Relay response exceeded the 32 MiB limit.")
            return {
                "status_code": int(response.status),
                "headers": _response_headers(response.headers, config),
                "body": _redact_local_secrets(response_body, config),
            }
    except urllib.error.HTTPError as exc:
        # Redirects arrive here because following them is disabled. Return the
        # original response; never contact the Location target.
        response_body = _read_response_body(exc)
        if response_body is None:
            return _error_response(502, "Relay response exceeded the 32 MiB limit.")
        return {
            "status_code": int(exc.code),
            "headers": _response_headers(exc.headers, config),
            "body": _redact_local_secrets(response_body, config),
        }
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        logger.warning("OpenAI-compatible relay connection failed: %s", type(exc).__name__)
        return _error_response(502, "Could not reach the approved provider endpoint.")
    except Exception as exc:  # noqa: BLE001 - never break the WebSocket loop
        logger.error(
            "Unexpected OpenAI-compatible relay failure: %s",
            type(exc).__name__,
            exc_info=True,
        )
        return _error_response(500, "The OpenAI-compatible relay failed unexpectedly.")
