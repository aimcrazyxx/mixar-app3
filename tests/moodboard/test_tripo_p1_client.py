# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json

import httpx
import pytest
from mixar.modules.moodboard.core.tripo_p1_client import TripoP1Client, TripoP1Error


def test_upload_create_poll_uses_official_p1_v3_schema():
    created = {}
    polls = {"count": 0}

    def handler(request):
        assert request.headers["authorization"] == "Bearer secret"
        if request.url.path.endswith("/files"):
            assert b"Content-Type: image/jpeg" in request.content
            view = "front" if b"front.jpg" in request.content else "left"
            return httpx.Response(
                200, json={"code": 0, "data": {"file_token": f"file-{view}"}}
            )
        if request.url.path.endswith("/generation/multiview-to-model"):
            created.update(json.loads(request.content))
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "task-1"}})
        polls["count"] += 1
        if polls["count"] == 1:
            return httpx.Response(200, json={"code": 0, "data": {"status": "running"}})
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "status": "success",
                    "output": {"model_url": "https://cdn/model.glb"},
                },
            },
        )

    client = TripoP1Client("secret", transport=httpx.MockTransport(handler))
    try:
        tokens = {
            "front": client.upload_image(b"\xff\xd8\xfffront", "front.png"),
            "left": client.upload_image(b"\xff\xd8\xffleft", "left.png"),
        }
        task_id = client.create_multiview_task(
            tokens,
            texture=False,
            pbr=True,
            face_limit=500,
            model_seed=7,
            texture_alignment="geometry",
            orientation="align_image",
        )
        url = client.wait_for_model(task_id, interval=0.01)
    finally:
        client.close()

    assert created == {
        "inputs": [{"front": "file-front"}, {"left": "file-left"}],
        "model": "P1-20260311",
        "texture": False,
        "pbr": False,
        "texture_alignment": "geometry",
        "orientation": "align_image",
        "face_limit": 500,
        "model_seed": 7,
    }
    assert url == "https://cdn/model.glb"


def test_front_and_second_view_are_required():
    client = TripoP1Client(
        "secret", transport=httpx.MockTransport(lambda request: None)
    )
    try:
        with pytest.raises(ValueError, match="Front"):
            client.create_multiview_task({"left": "file-left"})
        with pytest.raises(ValueError, match="at least one other"):
            client.create_multiview_task({"front": "file-front"})
    finally:
        client.close()


def test_upload_rejects_unsupported_image_bytes_before_request():
    requested = []

    def handler(request):
        requested.append(request)
        return httpx.Response(200, json={})

    client = TripoP1Client("secret", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError, match="PNG, JPEG, or WebP"):
            client.upload_image(b"not-an-image", "front.png")
    finally:
        client.close()

    assert requested == []


def test_upload_accepts_webp_and_uses_matching_content_type():
    seen = {}

    def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, json={"code": 0, "data": {"file_token": "webp"}})

    client = TripoP1Client("secret", transport=httpx.MockTransport(handler))
    try:
        token = client.upload_image(b"RIFF\x00\x00\x00\x00WEBPdata", "front.jpg")
    finally:
        client.close()

    assert token == "webp"
    assert b"front.webp" in seen["body"]
    assert b"Content-Type: image/webp" in seen["body"]


def test_poll_failure_is_user_visible():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "status": "failed",
                    "error_code": 4221,
                    "error_message": "bad views",
                },
            },
        )

    client = TripoP1Client("secret", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(TripoP1Error, match="bad views"):
            client.wait_for_model("task", interval=0.01)
    finally:
        client.close()


def test_poll_rejects_unknown_status_instead_of_waiting_until_timeout():
    client = TripoP1Client(
        "secret",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"code": 0, "data": {"status": "mystery"}}
            )
        ),
    )
    try:
        with pytest.raises(TripoP1Error, match="unknown task status"):
            client.wait_for_model("task", interval=0.01)
    finally:
        client.close()


def test_poll_rejects_unsafe_model_download_url():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "status": "success",
                    "output": {"model_url": "file:///local/secret.glb"},
                },
            },
        )

    client = TripoP1Client("secret", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(TripoP1Error, match="invalid model download URL"):
            client.wait_for_model("task", interval=0.01)
    finally:
        client.close()


def test_http_error_does_not_echo_tripo_api_key():
    secret = "tripo-secret-123"

    client = TripoP1Client(
        secret,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                401, json={"message": f"Credential {secret} is invalid"}
            )
        ),
    )
    try:
        with pytest.raises(TripoP1Error) as error:
            client.upload_image(b"\xff\xd8\xffimage", "front.jpg")
    finally:
        client.close()

    assert secret not in str(error.value)
    assert "[REDACTED]" in str(error.value)


def test_api_error_body_does_not_echo_tripo_api_key():
    secret = "tripo-secret-456"
    client = TripoP1Client(
        secret,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"code": 1001, "message": f"Invalid {secret}"}
            )
        ),
    )
    try:
        with pytest.raises(TripoP1Error) as error:
            client.upload_image(b"\xff\xd8\xffimage", "front.jpg")
    finally:
        client.close()

    assert secret not in str(error.value)
    assert "[REDACTED]" in str(error.value)
