# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Minimal, defensive client for Tripo's public v3 generation API."""

from __future__ import annotations

import time
from urllib.parse import urlsplit

import httpx

TRIPO_BASE_URL = "https://openapi.tripo3d.ai/v3"
TRIPO_MODEL_STANDARD = "v3.1-20260211"
TRIPO_MODEL_P1 = "P1-20260311"
TRIPO_MODELS = frozenset({TRIPO_MODEL_STANDARD, TRIPO_MODEL_P1})

_VIEW_ORDER = ("front", "left", "back", "right")
_TERMINAL_FAILURES = frozenset(
    {"failed", "cancelled", "canceled", "banned", "expired"}
)


class TripoError(RuntimeError):
    """Safe, user-displayable Tripo error."""


def validate_api_key(api_key: str) -> str:
    value = str(api_key or "").strip()
    if not value:
        raise ValueError("Configure a Tripo API key first")
    if not value.startswith("tsk_"):
        raise ValueError("Tripo API keys must start with 'tsk_'")
    if len(value) > 256 or any(ch.isspace() for ch in value):
        raise ValueError("Invalid Tripo API key")
    return value


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


def _generation_options(
    *,
    texture=True,
    pbr=True,
    face_limit=0,
    model_seed=0,
    texture_quality="standard",
    geometry_quality="standard",
    texture_alignment=None,
    orientation=None,
) -> dict:
    quality_values = {"standard", "detailed"}
    if texture_quality not in quality_values:
        raise ValueError("Invalid Tripo texture quality")
    if geometry_quality not in quality_values:
        raise ValueError("Invalid Tripo geometry quality")
    if texture_alignment not in {"original_image", "geometry", None}:
        raise ValueError("Invalid Tripo texture alignment")
    if orientation not in {"default", "align_image", None}:
        raise ValueError("Invalid Tripo orientation")

    texture = bool(texture)
    result = {
        "texture": texture,
        "pbr": bool(pbr and texture),
        "texture_quality": texture_quality,
        "geometry_quality": geometry_quality,
    }
    if face_limit:
        limit = int(face_limit)
        if not 50 <= limit <= 20000:
            raise ValueError("Face limit must be between 50 and 20,000")
        result["face_limit"] = limit
    if model_seed:
        result["model_seed"] = int(model_seed)
    if texture_alignment is not None:
        result["texture_alignment"] = texture_alignment
    if orientation is not None:
        result["orientation"] = orientation
    return result


class TripoClient:
    def __init__(
        self,
        api_key: str,
        *,
        model=TRIPO_MODEL_STANDARD,
        timeout=120.0,
        transport=None,
    ):
        self._api_key = validate_api_key(api_key)
        if model not in TRIPO_MODELS:
            raise ValueError("Unsupported Tripo model version")
        self.model = model
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
        # Task-creation POSTs are intentionally never retried: a timed-out
        # response may still represent a billable task accepted by Tripo.
        attempts = 3 if method.upper() == "GET" else 1
        response = None
        for attempt in range(attempts):
            try:
                response = self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                if attempt + 1 < attempts:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise TripoError("Tripo request timed out") from exc
            except httpx.HTTPError as exc:
                if attempt + 1 < attempts:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise TripoError(
                    f"Could not reach Tripo: {type(exc).__name__}"
                ) from exc

            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt + 1 < attempts:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = min(max(float(retry_after), 0.05), 5.0)
                except ValueError:
                    delay = 0.5 * (attempt + 1)
                time.sleep(delay)

        if response is None:
            raise TripoError("Tripo request failed")
        if response.status_code >= 400:
            labels = {
                401: "Tripo rejected the API key",
                403: "Tripo access denied",
                429: "Tripo rate limit reached",
            }
            detail = self._safe_detail(_message(response))
            raise TripoError(
                f"{labels.get(response.status_code, 'Tripo request failed')}"
                f"{': ' + detail if detail else ''}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise TripoError("Tripo returned an invalid JSON response") from exc
        if not isinstance(body, dict):
            raise TripoError("Tripo returned an unexpected response")
        if body.get("code", 0) not in (0, None):
            raise TripoError(
                self._safe_detail(body.get("message") or "Tripo returned an error")
            )
        return body

    @staticmethod
    def _task_id(body: dict) -> str:
        task_id = (body.get("data") or {}).get("task_id")
        if not task_id:
            raise TripoError("Tripo response did not include task_id")
        return str(task_id)

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
            raise TripoError("Tripo upload response did not include file_token")
        return str(token)

    def create_text_task(self, prompt: str, **options) -> str:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("A prompt is required for Tripo text-to-model")
        if len(prompt) > 2048:
            raise ValueError("Tripo prompt exceeds 2,048 characters")
        payload = {"prompt": prompt, "model": self.model}
        payload.update(_generation_options(**options))
        return self._task_id(
            self._request("POST", "/generation/text-to-model", json=payload)
        )

    def create_image_task(self, token: str, **options) -> str:
        if not str(token or "").strip():
            raise ValueError("An uploaded image token is required")
        payload = {"input": str(token).strip(), "model": self.model}
        payload.update(
            _generation_options(
                texture_alignment=options.pop(
                    "texture_alignment", "original_image"
                ),
                orientation=options.pop("orientation", "default"),
                **options,
            )
        )
        return self._task_id(
            self._request("POST", "/generation/image-to-model", json=payload)
        )

    def create_multiview_task(self, tokens: dict[str, str], **options) -> str:
        clean = {
            key: str(tokens.get(key) or "").strip()
            for key in _VIEW_ORDER
            if str(tokens.get(key) or "").strip()
        }
        if "front" not in clean:
            raise ValueError("Front view is required")
        if len(clean) < 2:
            raise ValueError("Tripo requires front plus at least one other view")
        payload = {
            "files": [
                {"view": view, "file": {"file_token": clean[view]}}
                for view in _VIEW_ORDER
                if view in clean
            ],
            "model": self.model,
        }
        payload.update(
            _generation_options(
                texture_alignment=options.pop(
                    "texture_alignment", "original_image"
                ),
                orientation=options.pop("orientation", "default"),
                **options,
            )
        )
        return self._task_id(
            self._request("POST", "/generation/multiview-to-model", json=payload)
        )

    def wait_for_model(
        self,
        task_id: str,
        *,
        should_cancel=None,
        interval=2.0,
        max_wait=1800.0,
    ) -> str:
        task_id = str(task_id or "").strip()
        if not task_id or "/" in task_id or "?" in task_id:
            raise ValueError("Invalid Tripo task ID")
        deadline = time.monotonic() + max(1.0, float(max_wait))
        while True:
            if should_cancel and should_cancel():
                raise TripoError(
                    "Tripo generation cancelled locally; the remote task may continue"
                )
            if time.monotonic() >= deadline:
                raise TripoError("Tripo generation timed out after 30 minutes")
            body = self._request("GET", f"/tasks/{task_id}")
            data = body.get("data") or {}
            status = str(data.get("status") or "").lower()
            if status == "success":
                url = self._model_url(data)
                if not url:
                    raise TripoError("Tripo task succeeded without a model URL")
                return url
            if status in _TERMINAL_FAILURES:
                raise TripoError(
                    self._safe_detail(
                        data.get("error_message")
                        or data.get("error_msg")
                        or data.get("message")
                        or data.get("error")
                        or data.get("error_code")
                        or f"Tripo task {status}"
                    )
                )
            if status not in {"queued", "running"}:
                raise TripoError(
                    f"Tripo returned an unknown task status: {status or 'empty'}"
                )
            time.sleep(max(0.05, float(interval)))

    @staticmethod
    def _model_url(data: dict) -> str:
        output = data.get("output") or {}
        candidates = []
        if isinstance(output, dict):
            for key in ("model_url", "pbr_model", "model", "base_model"):
                value = output.get(key)
                if isinstance(value, str):
                    candidates.append(value)
                elif isinstance(value, dict):
                    candidates.extend(
                        item for item in value.values() if isinstance(item, str)
                    )
        for url in candidates:
            parts = urlsplit(url)
            if (
                parts.scheme == "https"
                and parts.netloc
                and parts.username is None
                and parts.password is None
            ):
                return url
        if candidates:
            raise TripoError("Tripo returned an invalid model download URL")
        return ""
