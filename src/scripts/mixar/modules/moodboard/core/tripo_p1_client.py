# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small Tripo v3 P1 multiview client matching the public API contract."""

from __future__ import annotations

import time
from urllib.parse import urlsplit

import httpx

TRIPO_BASE_URL = "https://openapi.tripo3d.ai/v3"
TRIPO_P1_MODEL = "P1-20260311"


class TripoP1Error(RuntimeError):
    pass


def _image_upload_metadata(image_bytes: bytes, filename: str) -> tuple[str, str]:
    """Return a filename/MIME pair that agrees with the actual bytes."""
    base = str(filename or "image").rsplit(".", 1)[0] or "image"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return f"{base}.png", "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return f"{base}.jpg", "image/jpeg"
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return f"{base}.webp", "image/webp"
    raise ValueError(f"{filename or 'image'} must contain PNG, JPEG, or WebP data")


def _message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:400]
    if not isinstance(data, dict):
        return str(data)[:400]
    error = data.get("error")
    if isinstance(error, dict):
        error = error.get("message") or error.get("code") or error
    return str(data.get("message") or error or "")[:400]


class TripoP1Client:
    def __init__(self, api_key: str, *, timeout=120.0, transport=None):
        self._api_key = str(api_key or "").strip()
        if not self._api_key:
            raise ValueError("Configure a Tripo API key first")
        self._client = httpx.Client(
            base_url=TRIPO_BASE_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(timeout, connect=min(timeout, 15.0)),
            transport=transport,
        )

    def close(self):
        self._client.close()

    def _safe_detail(self, value) -> str:
        return str(value or "").replace(self._api_key, "[REDACTED]")[:400]

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise TripoP1Error("Tripo request timed out") from exc
        except httpx.HTTPError as exc:
            raise TripoP1Error(f"Could not reach Tripo: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            labels = {
                401: "Tripo rejected the API key",
                403: "Tripo access denied",
                429: "Tripo rate limit reached",
            }
            detail = self._safe_detail(_message(response))
            raise TripoP1Error(
                f"{labels.get(response.status_code, 'Tripo request failed')}"
                f"{': ' + detail if detail else ''}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise TripoP1Error("Tripo returned an invalid JSON response") from exc
        if not isinstance(body, dict):
            raise TripoP1Error("Tripo returned an unexpected response")
        if isinstance(body, dict) and body.get("code", 0) not in (0, None):
            raise TripoP1Error(
                self._safe_detail(body.get("message") or "Tripo returned an error")
            )
        return body

    def upload_image(self, image_bytes: bytes, filename: str) -> str:
        if not image_bytes:
            raise ValueError(f"{filename} is empty")
        if len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError(f"{filename} exceeds Tripo's 20 MB image limit")
        upload_name, media_type = _image_upload_metadata(image_bytes, filename)
        body = self._request(
            "POST",
            "/files",
            files={"file": (upload_name, image_bytes, media_type)},
        )
        token = (body.get("data") or {}).get("file_token")
        if not token:
            raise TripoP1Error("Tripo upload response did not include file_token")
        return str(token)

    def create_multiview_task(
        self,
        tokens: dict[str, str],
        *,
        texture=True,
        pbr=True,
        face_limit=0,
        model_seed=0,
        texture_alignment="original_image",
        orientation="default",
    ) -> str:
        clean = {key: value for key, value in tokens.items() if value}
        if "front" not in clean:
            raise ValueError("Front view is required")
        if len(clean) < 2:
            raise ValueError("Tripo P1 requires front plus at least one other view")
        if texture_alignment not in {"original_image", "geometry"}:
            raise ValueError("Invalid Tripo texture alignment")
        if orientation not in {"default", "align_image"}:
            raise ValueError("Invalid Tripo orientation")
        texture = bool(texture)
        payload = {
            "inputs": [
                {view: clean[view]}
                for view in ("front", "left", "back", "right")
                if view in clean
            ],
            "model": TRIPO_P1_MODEL,
            "texture": texture,
            "pbr": bool(pbr and texture),
            "texture_alignment": texture_alignment,
            "orientation": orientation,
        }
        if face_limit:
            if not 50 <= int(face_limit) <= 20000:
                raise ValueError("Face limit must be between 50 and 20,000")
            payload["face_limit"] = int(face_limit)
        if model_seed:
            payload["model_seed"] = int(model_seed)
        body = self._request("POST", "/generation/multiview-to-model", json=payload)
        task_id = (body.get("data") or {}).get("task_id")
        if not task_id:
            raise TripoP1Error("Tripo response did not include task_id")
        return str(task_id)

    def wait_for_model(
        self,
        task_id: str,
        *,
        should_cancel=None,
        interval=2.0,
        max_wait=1800.0,
    ) -> str:
        deadline = time.monotonic() + max(1.0, float(max_wait))
        while True:
            if should_cancel and should_cancel():
                raise TripoP1Error("Tripo generation cancelled")
            if time.monotonic() >= deadline:
                raise TripoP1Error("Tripo generation timed out after 30 minutes")
            body = self._request("GET", f"/tasks/{task_id}")
            data = body.get("data") or {}
            status = str(data.get("status") or "").lower()
            if status == "success":
                url = (data.get("output") or {}).get("model_url")
                if not url:
                    raise TripoP1Error("Tripo task succeeded without a model URL")
                parts = urlsplit(str(url))
                if (
                    parts.scheme != "https"
                    or not parts.netloc
                    or parts.username is not None
                    or parts.password is not None
                ):
                    raise TripoP1Error("Tripo returned an invalid model download URL")
                return str(url)
            if status in {"failed", "cancelled", "banned"}:
                raise TripoP1Error(
                    self._safe_detail(
                        data.get("error_message")
                        or data.get("message")
                        or data.get("error")
                        or data.get("error_code")
                        or f"Tripo task {status}"
                    )
                )
            if status not in {"queued", "running"}:
                raise TripoP1Error(
                    f"Tripo returned an unknown task status: {status or 'empty'}"
                )
            time.sleep(max(0.05, float(interval)))
