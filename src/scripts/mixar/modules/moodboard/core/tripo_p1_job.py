# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Direct BYOK Tripo P1 job that rejoins Mixar's normal GLB import queue."""

import threading
from dataclasses import dataclass, field

from mixar.modules.common.api.response import APIResponse
from mixar.modules.common.job_queue import Job, JobState

from .tripo_p1_client import TripoP1Client, TripoP1Error


@dataclass
class TripoP1MultiViewJob(Job):
    service: str = "model_3d"
    model: str = "tripo-low"
    images: dict = field(default_factory=dict, repr=False)
    api_key: str = field(default="", repr=False)
    texture: bool = True
    pbr: bool = True
    face_limit: int = 0
    model_seed: int = 0
    texture_alignment: str = "original_image"
    orientation: str = "default"
    _model_url: str = field(default="", repr=False)

    def submit(self, on_success, on_error):
        # Snapshot before the worker starts. Cancellation releases the job's
        # large fields immediately, but must not mutate a dict mid-upload.
        image_snapshot = dict(self.images)
        key_snapshot = str(self.api_key)

        def _main(callback, value):
            import bpy

            def _run():
                callback(value)
                return None

            bpy.app.timers.register(_run, first_interval=0.0)

        def _worker():
            client = None
            try:
                client = TripoP1Client(key_snapshot)
                tokens = {}
                for view, data in image_snapshot.items():
                    if not data:
                        continue
                    if self.state == JobState.CANCELLED:
                        raise TripoP1Error("Tripo generation cancelled")
                    tokens[view] = client.upload_image(data, f"{view}.jpg")
                if self.state == JobState.CANCELLED:
                    raise TripoP1Error("Tripo generation cancelled")
                task_id = client.create_multiview_task(
                    tokens,
                    texture=self.texture,
                    pbr=self.pbr,
                    face_limit=self.face_limit,
                    model_seed=self.model_seed,
                    texture_alignment=self.texture_alignment,
                    orientation=self.orientation,
                )
                self._model_url = client.wait_for_model(
                    task_id,
                    should_cancel=lambda: self.state == JobState.CANCELLED,
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

        threading.Thread(target=_worker, daemon=True, name="MixarTripoP1").start()

    def parse_submit_response(self, response):
        if not self._model_url:
            raise ValueError("Tripo P1 returned no model URL")
        self.images = {}
        self.api_key = ""

    def parse_poll_response(self, response):
        return "DONE", self.inline_result_files()

    def should_skip_poll(self):
        return bool(self._model_url)

    def inline_result_files(self):
        return [{"type": "GLB", "url": self._model_url}] if self._model_url else []

    def release_resources(self):
        self.images = {}
        self.api_key = ""
