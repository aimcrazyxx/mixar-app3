# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small Tripo v3 P1 multiview client matching the public API contract."""

from __future__ import annotations

import time

import httpx

TRIPO_BASE_URL = "https://openapi.tripo3d.ai/v3"
TRIPO_P1_MODEL = "P1-20260311"


class TripoP1Error(RuntimeError):
    pass


def _message(response: httpx.Response) -> str:
    try:
        data = response.json()
        return str(data.get("message") or data.get("error") or "")[:400]
    except Exception:
        return response.text[:400]


class TripoP1Client:
    def __init__(self, api_key: str, *, timeout=120.0, transport=None):
        if not str(api_key or "").strip():
            raise ValueError("Configure a Tripo API key first")
        self._client = httpx.Client(
            base_url=TRIPO_BASE_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=httpx.Timeout(timeout, connect=min(timeout, 15.0)),
            transport=transport,
        )

    def close(self):
        self._client.close()

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
            detail = _message(response)
            raise TripoP1Error(
                f"{labels.get(response.status_code, 'Tripo request failed')}"
                f"{': ' + detail if detail else ''}"
            )
        body = response.json()
        if isinstance(body, dict) and body.get("code", 0) not in (0, None):
            raise TripoP1Error(str(body.get("message") or "Tripo returned an error"))
        return body

    def upload_image(self, image_bytes: bytes, filename: str) -> str:
        if not image_bytes:
            raise ValueError(f"{filename} is empty")
        if len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError(f"{filename} exceeds Tripo's 20 MB image limit")
        body = self._request(
            "POST", "/files",
            files={"file": (filename, image_bytes, "image/png")},
        )
        token = (body.get("data") or {}).get("file_token")
        if not token:
            raise TripoP1Error("Tripo upload response did not include file_token")
        return str(token)

    def create_multiview_task(
        self, tokens: dict[str, str], *, texture=True, pbr=True,
        face_limit=0, model_seed=0,
    ) -> str:
        clean = {key: value for key, value in tokens.items() if value}
        if "front" not in clean:
            raise ValueError("Front view is required")
        if len(clean) < 2:
            raise ValueError("Tripo P1 requires front plus at least one other view")
        payload = {
            "inputs": [{view: clean[view]} for view in ("front", "left", "back", "right") if view in clean],
            "model": TRIPO_P1_MODEL,
            "texture": bool(texture or pbr),
            "pbr": bool(pbr),
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
        self, task_id: str, *, should_cancel=None, interval=2.0, max_wait=1800.0,
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
                return str(url)
            if status in {"failed", "cancelled", "unknown"}:
                raise TripoP1Error(str(data.get("message") or data.get("error") or f"Tripo task {status}"))
            time.sleep(max(0.05, float(interval)))
