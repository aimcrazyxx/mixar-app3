# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json

import httpx
import pytest

from mixar.modules.moodboard.core.tripo_client import (
    TRIPO_MODEL_STANDARD,
    TripoClient,
)


def test_standard_text_to_model_uses_v3_endpoint_and_model_field():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": 0, "data": {"task_id": "text-1"}})

    client = TripoClient("tsk_test", transport=httpx.MockTransport(handler))
    try:
        assert client.create_text_task("a small robot", texture=False) == "text-1"
    finally:
        client.close()

    assert seen["path"].endswith("/v3/generation/text-to-model")
    assert seen["body"]["model"] == TRIPO_MODEL_STANDARD
    assert seen["body"]["prompt"] == "a small robot"
    assert "model_version" not in seen["body"]


def test_standard_image_to_model_uses_uploaded_token_as_input():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"code": 0, "data": {"task_id": "image-1"}})

    client = TripoClient("tsk_test", transport=httpx.MockTransport(handler))
    try:
        assert client.create_image_task("file_123") == "image-1"
    finally:
        client.close()

    assert seen["path"].endswith("/v3/generation/image-to-model")
    assert seen["body"]["input"] == "file_123"
    assert seen["body"]["model"] == TRIPO_MODEL_STANDARD


def test_api_key_shape_is_validated_before_network_access():
    with pytest.raises(ValueError, match="tsk_"):
        TripoClient("not-a-tripo-key")


def test_task_creation_post_is_not_retried_after_server_error():
    calls = {"count": 0}

    def handler(_request):
        calls["count"] += 1
        return httpx.Response(503, json={"message": "temporarily unavailable"})

    client = TripoClient("tsk_test", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(Exception, match="Tripo request failed"):
            client.create_text_task("a chair")
    finally:
        client.close()

    assert calls["count"] == 1
