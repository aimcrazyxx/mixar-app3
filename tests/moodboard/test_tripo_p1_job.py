# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_job_module(monkeypatch):
    class JobState(str, Enum):
        PENDING = "PENDING"
        CANCELLED = "CANCELLED"

    @dataclass
    class Job:
        feature_key: str = ""
        label: str = ""
        state: JobState = JobState.PENDING

    class APIResponse:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    response_module = ModuleType("mixar.modules.common.api.response")
    response_module.APIResponse = APIResponse
    queue_module = ModuleType("mixar.modules.common.job_queue")
    queue_module.Job = Job
    queue_module.JobState = JobState
    monkeypatch.setitem(sys.modules, response_module.__name__, response_module)
    monkeypatch.setitem(sys.modules, queue_module.__name__, queue_module)

    path = (
        Path(__file__).parents[2]
        / "src/scripts/mixar/modules/moodboard/core/tripo_p1_job.py"
    )
    name = "mixar.modules.moodboard.core._tripo_p1_job_test_target"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, JobState


def test_p1_job_returns_glb_to_the_standard_download_import_lane(monkeypatch):
    module, _job_state = _load_job_module(monkeypatch)
    job = module.TripoP1MultiViewJob(feature_key="model_3d")
    job._model_url = "https://cdn.example.test/result.glb"

    assert job.should_skip_poll() is True
    assert job.inline_result_files() == [
        {"type": "GLB", "url": "https://cdn.example.test/result.glb"}
    ]
    assert job.parse_poll_response(None) == (
        "DONE",
        [{"type": "GLB", "url": "https://cdn.example.test/result.glb"}],
    )


def test_cancelled_p1_job_never_uploads_or_creates_a_billable_task(
    monkeypatch,
):
    module, job_state = _load_job_module(monkeypatch)
    calls = []
    completed = threading.Event()
    errors = []

    class FakeClient:
        def __init__(self, _key):
            calls.append("client")

        def upload_image(self, *_args, **_kwargs):
            calls.append("upload")

        def create_multiview_task(self, *_args, **_kwargs):
            calls.append("create")

        def close(self):
            calls.append("close")

    bpy = ModuleType("bpy")
    bpy.app = SimpleNamespace(
        timers=SimpleNamespace(register=lambda callback, **_kwargs: callback())
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(module, "TripoP1Client", FakeClient)

    job = module.TripoP1MultiViewJob(
        feature_key="model_3d",
        images={"front": b"front", "left": b"left"},
        api_key="secret",
    )
    job.state = job_state.CANCELLED
    job.submit(
        lambda _response: completed.set(),
        lambda error: (errors.append(error), completed.set()),
    )

    assert completed.wait(1.0)
    assert calls == ["client", "close"]
    assert len(errors) == 1
    assert "cancelled" in str(errors[0]).lower()


def test_single_image_mode_does_not_enter_the_direct_p1_branch():
    source = (
        Path(__file__).parents[2]
        / "src/scripts/mixar/modules/moodboard/ui/operators/model_gen_ops.py"
    ).read_text(encoding="utf-8")

    assert "getattr(tab, 'tripo_input_mode', 'SINGLE') == 'MULTI'" in source
    assert "return self._execute_tripo_p1(context, tab, model)" in source
