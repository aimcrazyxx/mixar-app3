# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Direct Tripo BYOK generation that rejoins Mixar's GLB import queue."""

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from mixar.config.logging_config import get_logger
from mixar.modules.common.api.response import APIResponse
from mixar.modules.common.job_queue import Job, JobState

from .tripo_client import TripoClient, TripoError

logger = get_logger(__name__)


@dataclass
class TripoDirectGenerationJob(Job):
    service: str = "model_3d"
    model: str = "tripo-low"
    input_mode: str = "SINGLE"
    api_model: str = "v3.1-20260211"
    images: dict = field(default_factory=dict, repr=False)
    prompt: str = ""
    api_key: str = field(default="", repr=False)
    texture: bool = True
    pbr: bool = True
    face_limit: int = 0
    model_seed: int = 0
    texture_quality: str = "standard"
    geometry_quality: str = "standard"
    texture_alignment: str = "original_image"
    orientation: str = "default"
    _on_imported_hook: Optional[Callable] = field(default=None, repr=False)
    _model_url: str = field(default="", repr=False)
    provider_task_id: str = ""
    max_submit_attempts: int = 1

    def _options(self) -> dict:
        return {
            "texture": self.texture,
            "pbr": self.pbr,
            "face_limit": self.face_limit,
            "model_seed": self.model_seed,
            "texture_quality": self.texture_quality,
            "geometry_quality": self.geometry_quality,
            "texture_alignment": self.texture_alignment,
            "orientation": self.orientation,
        }

    def submit(self, on_success, on_error):
        # Snapshot secrets and image data before the worker starts. Terminal
        # cleanup can then release the job fields without mutating its upload.
        image_snapshot = dict(self.images)
        key_snapshot = str(self.api_key)
        prompt_snapshot = str(self.prompt)
        input_mode = str(self.input_mode or "SINGLE").upper()
        api_model = str(self.api_model)
        options = self._options()

        def _main(callback, value):
            import bpy

            def _run():
                callback(value)
                return None

            bpy.app.timers.register(_run, first_interval=0.0)

        def _cancelled() -> bool:
            return self.state == JobState.CANCELLED

        def _worker():
            client = None
            try:
                if _cancelled():
                    raise TripoError("Tripo generation cancelled locally")
                client = TripoClient(key_snapshot, model=api_model)

                if input_mode == "TEXT":
                    text_options = dict(options)
                    text_options.pop("texture_alignment", None)
                    text_options.pop("orientation", None)
                    task_id = client.create_text_task(
                        prompt_snapshot, **text_options
                    )
                elif input_mode == "SINGLE":
                    data = image_snapshot.get("front")
                    if not data:
                        raise ValueError("An image is required for Tripo image-to-model")
                    token = client.upload_image(data, "image.png")
                    if _cancelled():
                        raise TripoError("Tripo generation cancelled locally")
                    task_id = client.create_image_task(token, **options)
                elif input_mode == "MULTI":
                    tokens = {}
                    for view in ("front", "left", "back", "right"):
                        data = image_snapshot.get(view)
                        if not data:
                            continue
                        if _cancelled():
                            raise TripoError("Tripo generation cancelled locally")
                        tokens[view] = client.upload_image(data, f"{view}.png")
                    if _cancelled():
                        raise TripoError("Tripo generation cancelled locally")
                    task_id = client.create_multiview_task(tokens, **options)
                else:
                    raise ValueError("Invalid Tripo input mode")

                self.provider_task_id = task_id
                self._model_url = client.wait_for_model(
                    task_id, should_cancel=_cancelled
                )
                response = APIResponse(
                    success=True,
                    status_code=200,
                    data={"data": {"status": "DONE", "task_id": task_id}},
                )
                _main(on_success, response)
            except Exception as exc:
                _main(on_error, exc)
            finally:
                if client is not None:
                    client.close()

        threading.Thread(
            target=_worker,
            daemon=True,
            name="MixarTripoDirect",
        ).start()

    def parse_submit_response(self, response):
        if not self._model_url:
            raise ValueError("Tripo returned no model URL")
        self.images = {}
        self.api_key = ""

    def parse_poll_response(self, response):
        return "DONE", self.inline_result_files()

    def should_skip_poll(self):
        return bool(self._model_url)

    def inline_result_files(self):
        return [{"type": "GLB", "url": self._model_url}] if self._model_url else []

    def on_imported(self, object_names: str) -> None:
        super().on_imported(object_names)
        if self._on_imported_hook is not None:
            try:
                self._on_imported_hook(self, object_names)
            except Exception as exc:
                logger.warning("Tripo on_imported hook failed: %s", exc)

    def release_resources(self):
        self.images = {}
        self.api_key = ""


def enqueue_tripo_direct_job(
    *,
    feature_key: str,
    label: str,
    scene_flag: str = "",
    batch_popup_title: str = "",
    **kwargs,
):
    from mixar.modules.common.job_queue import (
        create_scene_flag_listener,
        get_queue,
        get_queue_with_listener,
    )

    job = TripoDirectGenerationJob(label=label, **kwargs)
    if scene_flag:
        listener = create_scene_flag_listener(
            scene_flag, batch_popup_title=batch_popup_title
        )
        queue = get_queue_with_listener(feature_key, listener)
    else:
        queue = get_queue(feature_key)
    if not queue.submit(job):
        return None
    return job
