# SPDX-FileCopyrightText: 2026 Mixar Authors
# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""WindowManager-attached properties for the BYOK dialog.

Split into three groups:
- Form fields (transient input for the Save flow)
- Cached display fields (mirror of server BYOK state, drives the gear
  icon and the "Currently in use" dialog section)
- Dialog state machine (drives what draw() renders)

The provider and model EnumProperty items both come from cache-backed
callbacks — see core/model_suggestions.py for the cache and fetch wiring.
Users see friendly labels; the IDs are what get sent to the server.
"""

import json

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from mixar.config.logging_config import get_logger

from ...constants import (
    BYOK_API_KEY_MAX_LENGTH,
    CODEX_DEFAULT_MODEL,
    DIALOG_STATE_ITEMS,
    OPENAI_COMPATIBLE_DEFAULT_BASE_URL,
    OPENAI_COMPATIBLE_DEFAULT_MODEL,
    OPENAI_COMPATIBLE_ROUTE_DIRECT,
    OPENAI_COMPATIBLE_ROUTE_MIXAR,
    OPENROUTER_DEFAULT_MODEL,
)
from ...core import custom_model_cache
from ...core.model_suggestions import get_model_items, get_provider_items

logger = get_logger(__name__)


def _provider_items(self, context):
    """EnumProperty items callback — provider list comes from the
    backend cache (populated by the models-catalog fetch on login).
    Returns a "Loading…" or "No providers configured" sentinel when
    the cache is empty so the dropdown is never blank.
    """
    return get_provider_items()


def _model_items(self, context):
    """EnumProperty items callback — model list filtered to the
    currently-selected provider. Returns a "No models available"
    sentinel when the provider has no cached models so the dropdown
    is never blank.
    """
    provider = getattr(context.window_manager, 'byok_form_provider', '') or ''
    return get_model_items(provider)


def _provider_changed(self, context):
    """When the user picks a different provider, reset the model
    selection so we don't end up with a model that doesn't belong to
    the new provider (which would be an invalid combo at save time).
    """
    try:
        wm = context.window_manager if context is not None else self
        provider = getattr(wm, 'byok_form_provider', '') or ''
        items = get_model_items(provider)
        # Dynamic EnumProperties reject identifiers absent from their current
        # item callback.  Select the first real model for the new provider, or
        # the NONE sentinel only when that is what the callback exposes.
        wm.byok_form_model = items[0][0] if items else 'NONE'
    except Exception as exc:
        logger.debug("Could not reset BYOK model selection: %s", type(exc).__name__)


def _custom_model_items(self, context):
    return custom_model_cache.get_items()


def _custom_discovered_changed(self, context):
    if self.byok_custom_discovered_model != '__manual__':
        self.byok_custom_model = self.byok_custom_discovered_model


def wipe_transient_secrets(wm) -> None:
    """Best-effort wipe of credential form fields on a WindowManager."""
    if wm is None:
        return
    for attr in (
        'byok_form_api_key', 'byok_form_codex_bundle',
        'byok_custom_api_key', 'byok_custom_api_key_visible',
    ):
        try:
            setattr(wm, attr, '')
        except Exception as exc:
            logger.debug(
                "Could not wipe transient BYOK field %s: %s",
                attr,
                type(exc).__name__,
            )


_WM_ATTRS = (
    'byok_form_provider',
    'byok_form_model',
    'byok_form_api_key',
    'byok_form_openrouter_model',
    'byok_form_codex_bundle',
    'byok_form_codex_model',
    'byok_is_active',
    'byok_current_provider',
    'byok_current_model',
    'byok_current_supports_vision',
    'byok_key_preview',
    'byok_dialog_state',
    'byok_last_error',
    'byok_custom_enabled',
    'byok_custom_route',
    'byok_custom_active_route',
    'byok_custom_base_url',
    'byok_custom_api_key',
    'byok_custom_api_key_visible',
    'byok_custom_show_key',
    'byok_custom_model',
    'byok_custom_discovered_model',
    'byok_custom_timeout',
    'byok_custom_max_output_tokens',
    'byok_custom_use_temperature',
    'byok_custom_temperature',
    'byok_custom_use_top_p',
    'byok_custom_top_p',
    'byok_custom_reasoning_effort',
    'byok_custom_headers',
    'byok_custom_context_limit',
    'byok_custom_tool_calling',
    'byok_custom_vision',
    'byok_custom_streaming',
    'byok_custom_endpoint_mode',
    'byok_custom_max_iterations',
    'byok_custom_advanced_expanded',
    'byok_custom_debug_enabled',
    'byok_custom_debug_expanded',
    'byok_custom_debug_report',
)


def register():
    WM = bpy.types.WindowManager

    # --- Form fields (what the user picks / types into the dialog) ---
    # Both provider and model use dynamic EnumProperty callbacks so the
    # lists can change without re-registering the properties. NOTE: when
    # items is a callback, EnumProperty does not accept a string default
    # — whichever item is index 0 in the cache becomes the default.
    WM.byok_form_provider = EnumProperty(
        name="Provider",
        description="LLM provider",
        items=_provider_items,
        update=_provider_changed,
    )
    WM.byok_form_model = EnumProperty(
        name="Model",
        description="Model to use with the selected provider",
        items=_model_items,
    )
    WM.byok_form_api_key = StringProperty(
        name="API Key",
        description=(
            "Your API key is stored encrypted and used only for Mixar agent requests. "
            "After saving, only a masked preview is shown."
        ),
        maxlen=BYOK_API_KEY_MAX_LENGTH,
        default='',
        subtype='PASSWORD',
        options={'SKIP_SAVE'},
    )

    # --- OpenRouter form field (shown when provider == 'openrouter'). The key
    # reuses byok_form_api_key; only the model is free-text and OpenRouter-specific.
    WM.byok_form_openrouter_model = StringProperty(
        name="Model",
        description="Any model slug from openrouter.ai/models (e.g. anthropic/claude-opus-4.8)",
        default=OPENROUTER_DEFAULT_MODEL,
    )

    # --- Codex form fields (shown when provider == 'codex') ---
    # The bundle is the full ~/.codex/auth.json (multi-KB, contains JWTs), so
    # a generous maxlen; PASSWORD hides the tokens (the Paste button + a char
    # count confirm it landed). Model is a free-text slug (lineup varies per
    # subscription tier).
    WM.byok_form_codex_bundle = StringProperty(
        name="Codex auth.json",
        description="Contents of ~/.codex/auth.json (run `codex login` first)",
        maxlen=16384,
        default='',
        subtype='PASSWORD',
        options={'SKIP_SAVE'},
    )
    WM.byok_form_codex_model = StringProperty(
        name="Model",
        description="Codex model slug, e.g. gpt-5.5 / gpt-5.4 / gpt-5.4-mini",
        default=CODEX_DEFAULT_MODEL,
    )

    # --- Cached display fields (mirror of server BYOK state) ---
    WM.byok_is_active = BoolProperty(
        name="BYOK Active",
        description="True when the server reports byok_active for this user",
        default=False,
    )
    WM.byok_current_provider = StringProperty(default='')
    WM.byok_current_model = StringProperty(default='')
    # Whether the saved model accepts image input. Text-only models (many
    # OpenRouter models) run chat fine but skip 3D visual feedback, so the
    # dialog surfaces a note. Defaults True (platform + vision models).
    WM.byok_current_supports_vision = BoolProperty(default=True)
    WM.byok_key_preview = StringProperty(default='')

    # --- Dialog state machine ---
    WM.byok_dialog_state = EnumProperty(
        name="Dialog State",
        items=DIALOG_STATE_ITEMS,
        default='IDLE',
    )
    WM.byok_last_error = StringProperty(default='')

    # --- OpenAI-compatible provider (non-secret live form state) ---
    WM.byok_custom_enabled = BoolProperty(default=False, options={'SKIP_SAVE'})
    WM.byok_custom_route = EnumProperty(
        name="Agent route",
        description="Choose which agent orchestrates requests to this endpoint",
        items=(
            (
                OPENAI_COMPATIBLE_ROUTE_MIXAR,
                "Mixar Orchestrator",
                "Use Mixar's full planner and Blender toolset through the secure device relay",
            ),
            (
                OPENAI_COMPATIBLE_ROUTE_DIRECT,
                "Direct Agent",
                "Run the lightweight compatible-provider agent inside Blender",
            ),
        ),
        default=OPENAI_COMPATIBLE_ROUTE_MIXAR,
    )
    # Saved route used by chat while the dialog's editable route changes.
    WM.byok_custom_active_route = StringProperty(default='', options={'SKIP_SAVE'})
    WM.byok_custom_base_url = StringProperty(
        name="Base URL", default=OPENAI_COMPATIBLE_DEFAULT_BASE_URL,
        description="API root; Mixar normalizes this to exactly one /v1",
    )
    WM.byok_custom_api_key = StringProperty(
        name="API Key", default='', maxlen=BYOK_API_KEY_MAX_LENGTH,
        subtype='PASSWORD', options={'SKIP_SAVE'},
    )
    WM.byok_custom_api_key_visible = StringProperty(
        name="API Key", default='', maxlen=BYOK_API_KEY_MAX_LENGTH,
        options={'SKIP_SAVE'},
    )
    WM.byok_custom_show_key = BoolProperty(default=False, options={'SKIP_SAVE'})
    WM.byok_custom_model = StringProperty(
        name="Model", default=OPENAI_COMPATIBLE_DEFAULT_MODEL,
    )
    WM.byok_custom_discovered_model = EnumProperty(
        name="Available models", items=_custom_model_items,
        update=_custom_discovered_changed,
    )
    WM.byok_custom_timeout = FloatProperty(
        name="Timeout", default=120.0, min=5.0, max=1800.0, subtype='TIME',
    )
    WM.byok_custom_max_output_tokens = IntProperty(
        name="Max output tokens", default=8192, min=1, max=1048576,
    )
    WM.byok_custom_use_temperature = BoolProperty(name="Temperature", default=False)
    WM.byok_custom_temperature = FloatProperty(default=0.7, min=0.0, max=2.0)
    WM.byok_custom_use_top_p = BoolProperty(name="Top P", default=False)
    WM.byok_custom_top_p = FloatProperty(default=1.0, min=0.0, max=1.0)
    WM.byok_custom_reasoning_effort = EnumProperty(
        name="Reasoning effort",
        items=(
            ('NONE', "Provider default", "Do not send reasoning_effort"),
            ('low', "Low", "Low reasoning effort"),
            ('medium', "Medium", "Medium reasoning effort"),
            ('high', "High", "High reasoning effort"),
            ('xhigh', "Extra high", "Extra-high reasoning effort"),
        ), default='NONE',
    )
    WM.byok_custom_headers = StringProperty(
        name="Custom headers (JSON)", default='', options={'SKIP_SAVE'},
        description='Optional JSON object, e.g. {"X-Organization":"team"}',
    )
    WM.byok_custom_context_limit = IntProperty(
        name="Context token limit", default=0, min=0, max=2097152,
        description="Approximate local trimming limit; 0 keeps all history",
    )
    WM.byok_custom_tool_calling = BoolProperty(name="Tool calling", default=True)
    WM.byok_custom_vision = BoolProperty(name="Vision", default=True)
    WM.byok_custom_streaming = BoolProperty(name="Streaming", default=True)
    WM.byok_custom_endpoint_mode = EnumProperty(
        name="API endpoint",
        items=(
            ('auto', "Auto", "Try Chat Completions, then Responses on 404/405"),
            ('chat_completions', "Chat Completions", "Use /chat/completions only"),
            ('responses', "Responses", "Use /responses only"),
        ), default='auto',
    )
    WM.byok_custom_max_iterations = IntProperty(
        name="Maximum agent iterations", default=20, min=1, max=100,
    )
    WM.byok_custom_advanced_expanded = BoolProperty(default=False)
    WM.byok_custom_debug_enabled = BoolProperty(name="Debug report", default=False)
    WM.byok_custom_debug_expanded = BoolProperty(default=False)
    WM.byok_custom_debug_report = StringProperty(
        default='', options={'SKIP_SAVE'}, maxlen=65535,
    )

    def _restore_custom_provider():
        try:
            from mixar.modules.common.secure_storage import get_secret, masked_preview
            wm = bpy.context.window_manager
            raw = get_secret('openai_compatible_config')
            if not raw:
                return None
            data = json.loads(raw)
            # Configs saved before the relay existed were direct-only. Keep
            # that proven route on upgrade; new configs default to Mixar.
            route = data.get('route', OPENAI_COMPATIBLE_ROUTE_DIRECT)
            if route not in {
                OPENAI_COMPATIBLE_ROUTE_MIXAR,
                OPENAI_COMPATIBLE_ROUTE_DIRECT,
            }:
                route = OPENAI_COMPATIBLE_ROUTE_DIRECT
            wm.byok_custom_route = route
            wm.byok_custom_active_route = route
            wm.byok_custom_base_url = data.get('base_url', wm.byok_custom_base_url)
            wm.byok_custom_model = data.get('model', wm.byok_custom_model)
            wm.byok_custom_timeout = data.get('timeout', wm.byok_custom_timeout)
            wm.byok_custom_max_output_tokens = data.get(
                'max_output_tokens', wm.byok_custom_max_output_tokens)
            wm.byok_custom_use_temperature = data.get('temperature') is not None
            if data.get('temperature') is not None:
                wm.byok_custom_temperature = data['temperature']
            wm.byok_custom_use_top_p = data.get('top_p') is not None
            if data.get('top_p') is not None:
                wm.byok_custom_top_p = data['top_p']
            wm.byok_custom_reasoning_effort = data.get('reasoning_effort') or 'NONE'
            wm.byok_custom_headers = json.dumps(data.get('custom_headers') or {})
            wm.byok_custom_context_limit = data.get('context_limit', 0)
            wm.byok_custom_tool_calling = data.get('tool_calling', True)
            wm.byok_custom_vision = data.get('vision', True)
            wm.byok_custom_streaming = data.get('streaming', True)
            wm.byok_custom_endpoint_mode = data.get('endpoint_mode', 'auto')
            wm.byok_custom_enabled = True
            wm.byok_is_active = True
            wm.byok_current_provider = 'openai-compatible'
            wm.byok_current_model = wm.byok_custom_model
            wm.byok_key_preview = masked_preview(
                get_secret('openai_compatible_api_key'))
        except Exception as exc:
            logger.warning(
                "Could not restore custom provider configuration: %s",
                type(exc).__name__,
            )
        return None

    bpy.app.timers.register(_restore_custom_provider, first_interval=0.0)


def unregister():
    WM = bpy.types.WindowManager
    # Clear live values before unregistering their RNA definitions.  This is
    # particularly important during Reload Scripts, where the WindowManager
    # instance survives module teardown.
    seen = set()
    candidates = []
    try:
        candidates.extend(list(bpy.data.window_managers))
    except Exception as exc:
        logger.debug("Could not enumerate WindowManagers: %s", type(exc).__name__)
    try:
        candidates.append(bpy.context.window_manager)
    except Exception as exc:
        logger.debug("Could not access active WindowManager: %s", type(exc).__name__)
    for wm in candidates:
        marker = id(wm)
        if marker not in seen:
            seen.add(marker)
            wipe_transient_secrets(wm)
    for attr in _WM_ATTRS:
        try:
            delattr(WM, attr)
        except AttributeError:
            logger.debug("BYOK property already absent during unregister: %s", attr)
