# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import sys
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

if "keyring" not in sys.modules:
    sys.modules["keyring"] = MagicMock()

from mixar.modules.byok.core import openai_compatible_relay as relay
from mixar.modules.byok.core.openai_compatible import OpenAICompatibleConfig
from mixar.modules.byok.ui.operators import openai_compatible_ops
from mixar.modules.common.api.services.agent_service import AgentService
from mixar.modules.space_mixie_chat.constants import JSONRPCMethod
from mixar.modules.space_mixie_chat.core import jsonrpc_client
from mixar.modules.space_mixie_chat.core.jsonrpc_client import JSONRPCWebSocketClient


def _secrets(*, route="MIXAR", key="device-secret", headers=None):
    config = {
        "route": route,
        "base_url": "http://127.0.0.1:11434/v1",
        "timeout": 42,
        "custom_headers": headers or {},
    }
    return {
        "openai_compatible_config": json.dumps(config),
        "openai_compatible_api_key": key,
    }


class _Response:
    def __init__(self, body=b'{"ok":true}', status=200, headers=None):
        self._body = body
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}

    def read(self, amount=-1):
        return self._body if amount < 0 else self._body[:amount]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_relay_injects_local_auth_and_ignores_backend_auth(monkeypatch):
    secrets = _secrets(headers={"X-Local-Key": "locally-approved"})
    captured = {}

    monkeypatch.setattr(relay, "get_secret", lambda name: secrets.get(name, ""))

    class _Opener:
        def open(self, request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _Response(
                body=b'{"echo":"device-secret locally-approved"}',
                headers={"Content-Type": "application/json", "Set-Cookie": "no"},
            )

    monkeypatch.setattr(relay, "_NO_REDIRECT_OPENER", _Opener())
    result = relay.handle_llm_request(
        {
            "method": "POST",
            "url": "http://127.0.0.1:11434/v1/chat/completions",
            "headers": {
                "Authorization": "Bearer backend-controlled",
                "Content-Type": "application/json",
                "X-Backend": "blocked",
            },
            "body": "{}",
            "timeout": 999,
        }
    )

    sent_headers = {name.lower(): value for name, value in captured["request"].header_items()}
    assert result["status_code"] == 200
    assert sent_headers["authorization"] == "Bearer device-secret"
    assert sent_headers["x-local-key"] == "locally-approved"
    assert "x-backend" not in sent_headers
    assert "set-cookie" not in {name.lower() for name in result["headers"]}
    assert "device-secret" not in result["body"]
    assert "locally-approved" not in result["body"]
    assert captured["timeout"] == 42


def test_relay_accepts_user_approved_custom_auth_without_api_key(monkeypatch):
    secrets = _secrets(key="", headers={"Authorization": "Token local-only"})
    captured = {}
    monkeypatch.setattr(relay, "get_secret", lambda name: secrets.get(name, ""))

    class _Opener:
        def open(self, request, timeout):
            captured["headers"] = dict(request.header_items())
            return _Response()

    monkeypatch.setattr(relay, "_NO_REDIRECT_OPENER", _Opener())
    result = relay.handle_llm_request(
        {
            "url": "http://127.0.0.1:11434/v1/responses",
            "method": "POST",
            "body": "{}",
        }
    )

    headers = {name.lower(): value for name, value in captured["headers"].items()}
    assert result["status_code"] == 200
    assert headers["authorization"] == "Token local-only"


def test_relay_rejects_unapproved_target_method_query_and_direct_route(monkeypatch):
    secrets = _secrets()
    monkeypatch.setattr(relay, "get_secret", lambda name: secrets.get(name, ""))

    for params in (
        {
            "method": "GET",
            "url": "http://127.0.0.1:11434/v1/chat/completions",
        },
        {
            "method": "POST",
            "url": "http://127.0.0.1:11435/v1/chat/completions",
        },
        {
            "method": "POST",
            "url": "http://127.0.0.1:11434/v1/chat/completions?next=evil",
        },
        {
            "method": "POST",
            "url": "http://127.0.0.1:11434/v1/models",
        },
    ):
        assert relay.handle_llm_request(params)["status_code"] == 403

    secrets.update(_secrets(route="DIRECT"))
    assert relay.handle_llm_request(
        {
            "method": "POST",
            "url": "http://127.0.0.1:11434/v1/chat/completions",
        }
    )["status_code"] == 403


def test_relay_enforces_request_and_response_limits(monkeypatch):
    secrets = _secrets()
    monkeypatch.setattr(relay, "get_secret", lambda name: secrets.get(name, ""))
    base = {
        "method": "POST",
        "url": "http://127.0.0.1:11434/v1/chat/completions",
    }

    oversized = dict(base, body=b"x" * (relay._MAX_REQUEST_BYTES + 1))
    assert relay.handle_llm_request(oversized)["status_code"] == 413

    class _Opener:
        def open(self, request, timeout):
            return _Response(body=b"x" * (relay._MAX_RESPONSE_BYTES + 1))

    monkeypatch.setattr(relay, "_NO_REDIRECT_OPENER", _Opener())
    assert relay.handle_llm_request(dict(base, body="{}"))["status_code"] == 502


def test_no_redirect_handler_never_follows_location():
    handler = relay._NoRedirect()
    assert handler.redirect_request(None, None, 307, "redirect", {}, "http://evil") is None


def test_agent_service_relay_payload_never_contains_api_key():
    captured = {}
    service = object.__new__(AgentService)
    service.put = lambda path, json: captured.update(path=path, payload=json)

    service.save_credentials_all(
        provider="local",
        model="qwen3",
        base_url="http://localhost:11434/v1",
    )

    assert captured["path"] == "byok"
    assert captured["payload"] == {
        "provider": "local",
        "model": "qwen3",
        "base_url": "http://localhost:11434/v1",
    }


def test_mixar_save_registers_relay_without_uploading_real_key(monkeypatch):
    wm = SimpleNamespace(byok_dialog_state="IDLE", byok_last_error="")
    config = OpenAICompatibleConfig(
        base_url="http://localhost:11434/v1",
        api_key="real-device-key",
        model="qwen3",
        endpoint_mode="auto",
    )
    secrets = {
        "openai_compatible_api_key": "old-key",
        "openai_compatible_config": "",
    }
    registered = {}
    scheduled = {}

    monkeypatch.setattr(openai_compatible_ops, "_key_value", lambda _wm: "real-device-key")
    monkeypatch.setattr(openai_compatible_ops, "_config", lambda _wm, _key: config)
    monkeypatch.setattr(openai_compatible_ops, "_route", lambda _wm: "MIXAR")
    monkeypatch.setattr(openai_compatible_ops, "_redraw", lambda: None)
    monkeypatch.setattr(
        openai_compatible_ops, "get_secret", lambda name: secrets.get(name, "")
    )

    def _set_secret(name, value):
        secrets[name] = value
        return True

    monkeypatch.setattr(openai_compatible_ops, "set_secret", _set_secret)
    monkeypatch.setattr(
        openai_compatible_ops.byok_client,
        "save_credentials_now",
        lambda **kwargs: (registered.update(kwargs) or True, {}, None),
    )
    monkeypatch.setattr(
        openai_compatible_ops.byok_client,
        "_schedule_on_main",
        lambda callback, *args: scheduled.update(callback=callback, args=args),
    )

    class _Provider:
        def __init__(self, provider_config):
            assert provider_config.endpoint_mode == "chat_completions"

        def test_connection(self):
            return None

        def close(self):
            return None

    class _ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(openai_compatible_ops, "OpenAICompatibleProvider", _Provider)
    monkeypatch.setattr(openai_compatible_ops.threading, "Thread", _ImmediateThread)

    assert openai_compatible_ops.start_request(wm, action="save") == {"FINISHED"}
    assert registered == {
        "provider": "local",
        "model": "qwen3",
        "base_url": "http://localhost:11434/v1",
    }
    locally_stored = json.loads(secrets["openai_compatible_config"])
    assert locally_stored["route"] == "MIXAR"
    assert "api_key" not in locally_stored
    assert secrets["openai_compatible_api_key"] == "real-device-key"
    assert scheduled["args"][3] is True


def test_jsonrpc_dispatches_llm_request():
    client = object.__new__(JSONRPCWebSocketClient)
    received = {}
    client._handle_llm_request = lambda params, request_id: received.update(
        params=params, request_id=request_id
    )

    client._handle_message(
        {
            "jsonrpc": "2.0",
            "id": "relay-1",
            "method": JSONRPCMethod.LLM_REQUEST,
            "params": {"url": "approved"},
        }
    )

    assert received == {"params": {"url": "approved"}, "request_id": "relay-1"}


def test_jsonrpc_relay_worker_queues_response(monkeypatch):
    client = object.__new__(JSONRPCWebSocketClient)
    client._outbound = Queue()
    expected = {"status_code": 200, "headers": {}, "body": "{}"}

    from mixar.modules.byok.core import openai_compatible_relay

    monkeypatch.setattr(
        openai_compatible_relay, "handle_llm_request", lambda _params: expected
    )

    class _ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(jsonrpc_client.threading, "Thread", _ImmediateThread)
    client._handle_llm_request({"body": "{}"}, "relay-2")

    assert json.loads(client._outbound.get_nowait()) == {
        "jsonrpc": "2.0",
        "id": "relay-2",
        "result": expected,
    }
