# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Non-blocking bridge from Mixie Chat to the local compatible-provider loop."""

from __future__ import annotations

import json
import threading
import uuid

from mixar.config.logging_config import get_logger
from mixar.modules.common.secure_storage import get_secret

from .agent_loop import SYSTEM_PROMPT, run_agent_loop
from .debug_report import to_json
from .openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    parse_custom_headers,
)
from .provider_types import ProviderError

logger = get_logger(__name__)

_cancel_events: dict[str, threading.Event] = {}
_active_runs: dict[str, str] = {}
_session_histories: dict[str, list[dict]] = {}
_stream_buffers: dict[str, str] = {}
_stream_update_pending: set[str] = set()


def is_active(wm=None) -> bool:
    try:
        if wm is None:
            import bpy

            wm = bpy.context.window_manager
        return bool(getattr(wm, "byok_custom_enabled", False))
    except Exception:
        return False


def cancel(scene_name: str) -> bool:
    event = _cancel_events.get(str(scene_name or ""))
    if event is None:
        return False
    event.set()
    return True


def _clear_run(scene_name: str, run_id: str) -> None:
    """Release only the run that still owns this scene's local-agent slot."""
    if _active_runs.get(scene_name) != run_id:
        return
    _active_runs.pop(scene_name, None)
    _cancel_events.pop(scene_name, None)
    _stream_buffers.pop(run_id, None)
    _stream_update_pending.discard(run_id)


def _snapshot_config(wm) -> OpenAICompatibleConfig:
    key = get_secret("openai_compatible_api_key")
    return OpenAICompatibleConfig(
        base_url=wm.byok_custom_base_url,
        api_key=key,
        model=wm.byok_custom_model,
        timeout=wm.byok_custom_timeout,
        max_output_tokens=wm.byok_custom_max_output_tokens,
        temperature=(
            wm.byok_custom_temperature if wm.byok_custom_use_temperature else None
        ),
        top_p=(wm.byok_custom_top_p if wm.byok_custom_use_top_p else None),
        reasoning_effort=wm.byok_custom_reasoning_effort
        if wm.byok_custom_reasoning_effort != "NONE"
        else "",
        custom_headers=parse_custom_headers(wm.byok_custom_headers),
        context_limit=wm.byok_custom_context_limit,
        tool_calling=wm.byok_custom_tool_calling,
        vision=wm.byok_custom_vision,
        streaming=wm.byok_custom_streaming,
        endpoint_mode=wm.byok_custom_endpoint_mode,
    )


def _scene_summary(scene) -> str:
    counts = {}
    names = []
    for obj in scene.objects:
        counts[obj.type] = counts.get(obj.type, 0) + 1
        if len(names) < 80:
            names.append(f"{obj.name} ({obj.type})")
    selected = [obj.name for obj in scene.objects if obj.select_get()]
    active = getattr(getattr(scene, "view_layers", None), "active", None)
    return json.dumps(
        {
            "scene": scene.name,
            "object_counts": counts,
            "objects": names,
            "selected": selected,
            "frame": scene.frame_current,
            "render_engine": scene.render.engine,
            "active_view_layer": getattr(active, "name", ""),
        },
        ensure_ascii=False,
    )


def _ui_history(scene) -> list[dict]:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in scene.mixie_chat_messages:
        if getattr(msg, "loader_visible", False) or str(
            getattr(msg, "bubble_id", "")
        ).startswith("temp_"):
            continue
        text = str(
            getattr(msg, "text", "") or getattr(msg, "content", "") or ""
        ).strip()
        if not text:
            continue
        role = "user" if getattr(msg, "sender", "") == "USER" else "assistant"
        history.append({"role": role, "content": text})
    return history


def _with_images(message: dict, encoded_attachments: list[dict]) -> dict:
    if not encoded_attachments:
        return message
    content = [{"type": "text", "text": str(message.get("content") or "")}]
    for item in encoded_attachments:
        mime = item.get("mime_type") or "image/png"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{item.get('base64', '')}"},
            }
        )
    return {**message, "content": content}


def _execute_tool_sync(script: str, timeout: float, cancel_event) -> dict:
    from mixar.modules.space_mixie_chat.core.executor import get_executor
    from mixar.modules.space_mixie_chat.core.main_thread_executor import (
        run_on_main_thread,
    )

    done = threading.Event()
    abandoned = threading.Event()
    holder = {}

    def _run():
        if abandoned.is_set() or cancel_event.is_set():
            done.set()
            return
        try:
            holder["result"] = get_executor().execute(script).to_dict()
        except Exception as exc:
            holder["result"] = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            done.set()

    run_on_main_thread(_run)
    while not done.wait(0.1):
        if cancel_event.is_set():
            abandoned.set()
            return {"success": False, "error": "Agent run cancelled"}
        timeout -= 0.1
        if timeout <= 0:
            abandoned.set()
            return {"success": False, "error": "Blender tool execution timed out"}
    return holder.get("result") or {
        "success": False,
        "error": "Tool returned no result",
    }


def start(scene, encoded_attachments=None, wire_message: str = "") -> tuple[bool, str]:
    """Snapshot Blender state on main thread, then run provider work in a daemon."""
    import bpy
    from mixar.modules.space_mixie_chat.core.executor import get_executor
    from mixar.modules.space_mixie_chat.core.session import get_session_manager

    wm = bpy.context.window_manager
    try:
        config = _snapshot_config(wm)
        max_iterations = int(wm.byok_custom_max_iterations)
    except Exception as exc:
        return False, str(exc)
    if not config.api_key and not config.custom_headers:
        return False, "Configure an API key or authentication header first"

    manager = get_session_manager()
    scene_name = scene.name
    if scene_name in _active_runs:
        return False, "The previous custom-provider run is still stopping"
    session_id = manager.start_session(scene, "custom-provider")
    try:
        existing = _session_histories.get(session_id)
        messages = list(existing) if existing else _ui_history(scene)
        if existing:
            latest = next(
                (m for m in reversed(_ui_history(scene)) if m.get("role") == "user"),
                None,
            )
            if latest:
                messages.append(latest)
        if wire_message:
            for index in range(len(messages) - 1, -1, -1):
                if messages[index].get("role") == "user":
                    messages[index] = {"role": "user", "content": wire_message}
                    break
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError("No user message was available for the custom provider")
        messages[0] = {
            "role": "system",
            "content": SYSTEM_PROMPT
            + "\n\nCURRENT BLENDER STATE (initial snapshot):\n"
            + _scene_summary(scene),
        }
        if config.vision and encoded_attachments:
            for index in range(len(messages) - 1, -1, -1):
                if messages[index].get("role") == "user":
                    messages[index] = _with_images(
                        messages[index], list(encoded_attachments)
                    )
                    break
    except Exception as exc:
        manager.set_connected(scene)
        logger.error("Could not build custom agent context: %s", type(exc).__name__)
        return False, "Could not build the custom-provider conversation context"

    run_id = uuid.uuid4().hex
    cancel_event = threading.Event()
    _cancel_events[scene_name] = cancel_event
    _active_runs[scene_name] = run_id
    _stream_buffers[run_id] = ""

    try:
        get_executor().begin_agent_turn()
    except Exception as exc:
        _clear_run(scene_name, run_id)
        manager.set_connected(scene)
        logger.error("Could not begin custom agent turn: %s", type(exc).__name__)
        return False, "Could not initialize the Blender agent executor"

    def _worker():
        provider = None
        try:
            provider = OpenAICompatibleProvider(config)
            result = run_agent_loop(
                provider,
                messages,
                lambda _name, args: _execute_tool_sync(
                    args["script"], config.timeout, cancel_event
                ),
                max_iterations=max_iterations,
                context_limit=config.context_limit,
                cancel_event=cancel_event,
                on_text_delta=lambda delta: _stream_delta(scene_name, run_id, delta),
            )
            _session_histories[session_id] = result.messages
            while len(_session_histories) > 32:
                _session_histories.pop(next(iter(_session_histories)))
            _finalize(scene_name, run_id, result.text, result.debug, None)
        except Exception as exc:
            _finalize(scene_name, run_id, "", {}, exc)
        finally:
            if provider is not None:
                try:
                    provider.close()
                except Exception as exc:
                    logger.debug(
                        "Could not close custom provider client: %s",
                        type(exc).__name__,
                    )

    threading.Thread(
        target=_worker,
        name=f"MixarCustomAgent-{uuid.uuid4().hex[:8]}",
        daemon=True,
    ).start()
    return True, ""


def _stream_delta(scene_name: str, run_id: str, delta: str) -> None:
    """Coalesce provider text deltas into the optimistic agent bubble."""
    from mixar.modules.space_mixie_chat.core.main_thread_executor import (
        run_on_main_thread,
    )

    if _active_runs.get(scene_name) != run_id:
        return
    _stream_buffers[run_id] = _stream_buffers.get(run_id, "") + str(delta)
    if run_id in _stream_update_pending:
        return
    _stream_update_pending.add(run_id)

    def _apply():
        import bpy
        from mixar.modules.space_mixie_chat.constants import TEMP_PLACEHOLDER_PREFIX
        from mixar.modules.space_mixie_chat.core.ui_utils import redraw_chat_areas

        _stream_update_pending.discard(run_id)
        if _active_runs.get(scene_name) != run_id:
            return
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            return
        text = _stream_buffers.get(run_id, "")
        for message in reversed(scene.mixie_chat_messages):
            if str(getattr(message, "bubble_id", "")).startswith(
                TEMP_PLACEHOLDER_PREFIX
            ):
                message.loader_visible = False
                if hasattr(message, "text"):
                    message.text = text
                elif hasattr(message, "content"):
                    message.content = text
                break
        redraw_chat_areas()

    run_on_main_thread(_apply)


def _finalize(scene_name: str, run_id: str, text: str, debug: dict, error) -> None:
    from mixar.modules.space_mixie_chat.core.main_thread_executor import (
        run_on_main_thread,
    )

    def _apply():
        import bpy
        from mixar.modules.space_mixie_chat.constants import (
            TEMP_PLACEHOLDER_PREFIX,
            SessionState,
        )
        from mixar.modules.space_mixie_chat.core.executor import get_executor
        from mixar.modules.space_mixie_chat.core.message_helpers import (
            add_agent_message,
        )
        from mixar.modules.space_mixie_chat.core.session import get_session_manager
        from mixar.modules.space_mixie_chat.core.ui_utils import redraw_chat_areas

        if _active_runs.get(scene_name) != run_id:
            return
        try:
            scene = bpy.data.scenes.get(scene_name)
            get_executor().end_agent_turn()
            if scene is None:
                return
            for index in range(len(scene.mixie_chat_messages) - 1, -1, -1):
                if str(
                    getattr(scene.mixie_chat_messages[index], "bubble_id", "")
                ).startswith(TEMP_PLACEHOLDER_PREFIX):
                    scene.mixie_chat_messages.remove(index)
            report = dict(debug)
            cancelled = (
                isinstance(error, ProviderError) and error.category == "cancelled"
            )
            if error is not None and not cancelled:
                message = "Custom provider request failed: " + str(error)
                add_agent_message(scene, message)
                report = {"error": str(error), **report}
            elif cancelled:
                report = {"cancelled": True, **report}
            else:
                add_agent_message(scene, text)
            wm = bpy.context.window_manager
            wm.byok_custom_debug_report = to_json(report)
            get_session_manager().set_state(scene, SessionState.IDLE)
            redraw_chat_areas()
        finally:
            _clear_run(scene_name, run_id)

    run_on_main_thread(_apply)
