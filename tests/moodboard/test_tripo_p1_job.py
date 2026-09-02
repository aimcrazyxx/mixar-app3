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
        service: str = ""
        origin_capability_key: str = ""
        model: str = ""
        state: JobState = JobState.PENDING

        def on_imported(self, object_names):
            self.imported_object_names = object_names

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
        / "src/scripts/mixar/modules/moodboard/core/tripo_direct_job.py"
    )
    name = "mixar.modules.moodboard.core._tripo_direct_job_test_target"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, JobState


def _immediate_bpy(monkeypatch):
    bpy = ModuleType("bpy")
    bpy.app = SimpleNamespace(
        timers=SimpleNamespace(register=lambda callback, **_kwargs: callback())
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)


def test_direct_job_returns_glb_to_standard_download_import_lane(monkeypatch):
    module, _job_state = _load_job_module(monkeypatch)
    job = module.TripoDirectGenerationJob(
        feature_key="model_3d",
        input_mode="MULTI",
        api_model="P1-20260311",
    )
    job._model_url = "https://cdn.example.test/result.glb"

    assert job.input_mode == "MULTI"
    assert job.api_model == "P1-20260311"
    assert job.should_skip_poll() is True
    assert job.inline_result_files() == [
        {"type": "GLB", "url": "https://cdn.example.test/result.glb"}
    ]


def test_cancelled_direct_job_never_creates_client_or_billable_task(monkeypatch):
    module, job_state = _load_job_module(monkeypatch)
    _immediate_bpy(monkeypatch)
    calls = []
    completed = threading.Event()
    errors = []

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            calls.append("client")

    monkeypatch.setattr(module, "TripoClient", FakeClient)
    job = module.TripoDirectGenerationJob(
        feature_key="model_3d",
        images={"front": b"front", "left": b"left"},
        api_key="tsk_secret",
        input_mode="MULTI",
    )
    job.state = job_state.CANCELLED
    job.submit(
        lambda _response: completed.set(),
        lambda error: (errors.append(error), completed.set()),
    )

    assert completed.wait(1.0)
    assert calls == []
    assert len(errors) == 1
    assert "cancelled" in str(errors[0]).lower()


def test_standard_text_job_uses_normal_model_and_returns_url(monkeypatch):
    module, _job_state = _load_job_module(monkeypatch)
    _immediate_bpy(monkeypatch)
    calls = []
    completed = threading.Event()
    responses = []

    class FakeClient:
        def __init__(self, _key, *, model):
            calls.append(("client", model))

        def create_text_task(self, prompt, **_options):
            calls.append(("text", prompt))
            return "task-1"

        def wait_for_model(self, task_id, **_kwargs):
            calls.append(("wait", task_id))
            return "https://cdn.example.test/normal.glb"

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(module, "TripoClient", FakeClient)
    job = module.TripoDirectGenerationJob(
        api_key="tsk_secret",
        api_model="v3.1-20260211",
        input_mode="TEXT",
        prompt="a chair",
    )
    job.submit(
        lambda response: (responses.append(response), completed.set()),
        lambda _error: completed.set(),
    )

    assert completed.wait(1.0)
    assert ("client", "v3.1-20260211") in calls
    assert ("text", "a chair") in calls
    assert job.provider_task_id == "task-1"
    assert job.inline_result_files()[0]["url"].endswith("normal.glb")
    assert len(responses) == 1


def test_operator_wires_optional_direct_route_and_backend_fallback():
    source = (
        Path(__file__).parents[2]
        / "src/scripts/mixar/modules/moodboard/ui/operators/model_gen_ops.py"
    ).read_text(encoding="utf-8")

    assert 'getattr(tab, "tripo_use_direct_api", False)' in source
    assert "return self._execute_tripo_direct" in source
    assert "return self._execute_tripo_multi_backend" in source
