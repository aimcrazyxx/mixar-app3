# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Async operators for the direct OpenAI-compatible provider."""

import json
import threading

import bpy
from bpy.types import Operator
from mixar.config.logging_config import get_logger
from mixar.modules.common.secure_storage import (
    delete_secret,
    get_secret,
    masked_preview,
    set_secret,
)

from ...core import byok_client
from ...core.custom_model_cache import replace as replace_custom_models
from ...core.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    parse_custom_headers,
)

logger = get_logger(__name__)


def _redraw():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    except Exception as exc:
        logger.debug("Could not redraw custom-provider UI: %s", type(exc).__name__)


def _key_value(wm) -> str:
    field = (
        "byok_custom_api_key_visible"
        if wm.byok_custom_show_key
        else "byok_custom_api_key"
    )
    return getattr(wm, field, "").strip() or get_secret("openai_compatible_api_key")


def _config(wm, key: str) -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        base_url=wm.byok_custom_base_url,
        api_key=key,
        model=wm.byok_custom_model,
        timeout=wm.byok_custom_timeout,
        max_output_tokens=wm.byok_custom_max_output_tokens,
        temperature=wm.byok_custom_temperature
        if wm.byok_custom_use_temperature
        else None,
        top_p=wm.byok_custom_top_p if wm.byok_custom_use_top_p else None,
        reasoning_effort=(
            wm.byok_custom_reasoning_effort
            if wm.byok_custom_reasoning_effort != "NONE"
            else ""
        ),
        custom_headers=parse_custom_headers(wm.byok_custom_headers),
        context_limit=wm.byok_custom_context_limit,
        tool_calling=wm.byok_custom_tool_calling,
        vision=wm.byok_custom_vision,
        streaming=wm.byok_custom_streaming,
        endpoint_mode=wm.byok_custom_endpoint_mode,
    )


def start_request(wm, *, action: str):
    try:
        key = _key_value(wm)
        config = _config(wm, key)
        if not key and not config.custom_headers:
            raise ValueError("Enter an API key or a custom authentication header")
    except Exception as exc:
        wm.byok_dialog_state = "ERROR"
        wm.byok_last_error = str(exc)
        return {"CANCELLED"}
    wm.byok_dialog_state = "SAVING"
    wm.byok_last_error = ""
    _redraw()

    def _worker():
        provider = None
        previous_key = (
            get_secret("openai_compatible_api_key") if action == "save" else ""
        )
        key_changed = False
        try:
            provider = OpenAICompatibleProvider(config)
            models = provider.list_models() if action == "models" else None
            if action != "models":
                provider.test_connection()
            if (
                action == "save"
                and key
                and not set_secret("openai_compatible_api_key", key)
            ):
                raise RuntimeError(
                    "The operating-system credential store rejected the API key"
                )
            key_changed = action == "save" and bool(key) and key != previous_key
            if action == "save":
                stored = {
                    name: getattr(config, name)
                    for name in (
                        "base_url",
                        "model",
                        "timeout",
                        "max_output_tokens",
                        "temperature",
                        "top_p",
                        "reasoning_effort",
                        "custom_headers",
                        "context_limit",
                        "tool_calling",
                        "vision",
                        "streaming",
                        "endpoint_mode",
                    )
                }
                if not set_secret("openai_compatible_config", json.dumps(stored)):
                    raise RuntimeError("Could not persist the custom provider settings")
            success, error = True, ""
        except Exception as exc:
            if key_changed:
                restored = (
                    set_secret("openai_compatible_api_key", previous_key)
                    if previous_key
                    else delete_secret("openai_compatible_api_key")
                )
                if not restored:
                    error = f"{exc}. The previous API key could not be restored"
                else:
                    error = str(exc)
            else:
                error = str(exc)
            models, success = None, False
        finally:
            if provider is not None:
                provider.close()
        byok_client._schedule_on_main(_done, action, config, success, models, error)

    threading.Thread(
        target=_worker,
        daemon=True,
        name=f"MixarCustomProvider-{action}",
    ).start()
    return {"FINISHED"}


def _done(action, config, success, models, error):
    wm = bpy.context.window_manager
    if not success:
        wm.byok_dialog_state = "ERROR"
        wm.byok_last_error = error or "Provider request failed"
        _redraw()
        return
    if action == "models":
        replace_custom_models(models or [])
        if models:
            wm.byok_custom_discovered_model = models[0]
    if action == "save":
        wm.byok_custom_enabled = True
        wm.byok_custom_base_url = config.base_url
        wm.byok_is_active = True
        wm.byok_current_provider = "openai-compatible"
        wm.byok_current_model = config.model
        wm.byok_current_supports_vision = config.vision
        wm.byok_key_preview = masked_preview(get_secret("openai_compatible_api_key"))
        wm.byok_custom_api_key = ""
        wm.byok_custom_api_key_visible = ""
    wm.byok_dialog_state = "IDLE"
    wm.byok_last_error = ""
    _redraw()


class MIXAR_BYOK_OT_custom_toggle_key(Operator):
    bl_idname = "mixar_byok.custom_toggle_key"
    bl_label = "Show or hide API key"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        wm = context.window_manager
        if wm.byok_custom_show_key:
            wm.byok_custom_api_key = wm.byok_custom_api_key_visible
        else:
            wm.byok_custom_api_key_visible = wm.byok_custom_api_key
        wm.byok_custom_show_key = not wm.byok_custom_show_key
        return {"FINISHED"}


class MIXAR_BYOK_OT_custom_test(Operator):
    bl_idname = "mixar_byok.custom_test"
    bl_label = "Test connection"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        return start_request(context.window_manager, action="test")


class MIXAR_BYOK_OT_custom_fetch_models(Operator):
    bl_idname = "mixar_byok.custom_fetch_models"
    bl_label = "Fetch model list"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        return start_request(context.window_manager, action="models")


class MIXAR_BYOK_OT_custom_copy_debug(Operator):
    bl_idname = "mixar_byok.custom_copy_debug"
    bl_label = "Copy sanitized debug report"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        context.window_manager.clipboard = (
            context.window_manager.byok_custom_debug_report
        )
        self.report({"INFO"}, "Sanitized debug report copied")
        return {"FINISHED"}


class MIXAR_BYOK_OT_custom_view_debug(Operator):
    bl_idname = "mixar_byok.custom_view_debug"
    bl_label = "AI Agent Debug"
    bl_options = {"INTERNAL"}

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=720)

    def draw(self, context):
        raw = context.window_manager.byok_custom_debug_report
        try:
            report = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            report = {"error": "The debug report could not be decoded"}
        fields = (
            ("provider", "Provider"),
            ("route", "Agent route"),
            ("model", "Model"),
            ("base_url", "Base URL"),
            ("endpoint", "Endpoint"),
            ("context_tokens_approx", "Approx. context tokens"),
            ("message_count", "Messages"),
            ("tool_count", "Available tools"),
            ("request_count", "Provider requests"),
            ("iterations", "Agent iterations"),
            ("duration_ms", "Duration (ms)"),
            ("http_status", "HTTP status"),
            ("finish_reason", "Finish reason"),
            ("reasoning_present", "Reasoning preserved"),
        )
        col = self.layout.column(align=True)
        for key, label in fields:
            row = col.row()
            row.label(text=label)
            row.label(text=str(report.get(key, "")))
        for key, label in (
            ("capability_warnings", "Capability notices"),
            ("degraded_parameters", "Capability fallbacks"),
            ("usage", "Usage"),
            ("tool_calls", "Tool calls"),
            ("tool_results", "Tool results"),
            ("error", "Error"),
        ):
            value = report.get(key)
            if value not in (None, "", [], {}):
                box = col.box()
                box.label(text=label)
                box.label(text=json.dumps(value, ensure_ascii=False)[:1000])

    def execute(self, _context):
        return {"FINISHED"}


classes = (
    MIXAR_BYOK_OT_custom_toggle_key,
    MIXAR_BYOK_OT_custom_test,
    MIXAR_BYOK_OT_custom_fetch_models,
    MIXAR_BYOK_OT_custom_view_debug,
    MIXAR_BYOK_OT_custom_copy_debug,
)
