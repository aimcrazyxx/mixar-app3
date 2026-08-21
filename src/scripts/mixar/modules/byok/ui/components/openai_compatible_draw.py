# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenAI-compatible settings section for the BYOK dialog."""


def draw(box, col, wm, draw_tall_prop):
    draw_tall_prop(col, wm, 'byok_custom_base_url', "Base URL")
    label = col.row(); label.enabled = False; label.label(text="API Key")
    key_row = col.row(align=True); key_row.scale_y = 1.35
    key_prop = 'byok_custom_api_key_visible' if wm.byok_custom_show_key else 'byok_custom_api_key'
    key_row.prop(wm, key_prop, text="")
    key_row.operator(
        "mixar_byok.custom_toggle_key", text="",
        icon='HIDE_OFF' if wm.byok_custom_show_key else 'HIDE_ON',
    )
    col.separator(factor=0.35)
    model_row = col.row(align=True); model_row.scale_y = 1.3
    model_row.prop(wm, 'byok_custom_discovered_model', text="Models")
    model_row.operator("mixar_byok.custom_fetch_models", text="", icon='FILE_REFRESH')
    draw_tall_prop(col, wm, 'byok_custom_model', "Model ID")
    test = col.row(); test.scale_y = 1.2
    test.operator("mixar_byok.custom_test", text="Test connection", icon='CHECKMARK')

    box.separator(factor=0.45)
    advanced = box.row()
    advanced.prop(
        wm, 'byok_custom_advanced_expanded', text="Advanced settings",
        icon='TRIA_DOWN' if wm.byok_custom_advanced_expanded else 'TRIA_RIGHT',
        emboss=False,
    )
    if wm.byok_custom_advanced_expanded:
        adv = box.column(align=True)
        adv.prop(wm, 'byok_custom_timeout')
        adv.prop(wm, 'byok_custom_endpoint_mode')
        adv.prop(wm, 'byok_custom_max_output_tokens')
        row = adv.row(align=True); row.prop(wm, 'byok_custom_use_temperature', text="")
        field = row.row(); field.enabled = wm.byok_custom_use_temperature
        field.prop(wm, 'byok_custom_temperature')
        row = adv.row(align=True); row.prop(wm, 'byok_custom_use_top_p', text="")
        field = row.row(); field.enabled = wm.byok_custom_use_top_p
        field.prop(wm, 'byok_custom_top_p')
        adv.prop(wm, 'byok_custom_reasoning_effort')
        adv.prop(wm, 'byok_custom_headers')
        adv.prop(wm, 'byok_custom_context_limit')
        adv.prop(wm, 'byok_custom_max_iterations')
        caps = adv.row(align=True)
        caps.prop(wm, 'byok_custom_tool_calling')
        caps.prop(wm, 'byok_custom_vision')
        caps.prop(wm, 'byok_custom_streaming')
        adv.prop(wm, 'byok_custom_debug_enabled')
        if wm.byok_custom_debug_enabled and wm.byok_custom_debug_report:
            adv.operator("mixar_byok.custom_copy_debug", icon='COPYDOWN')
    hint = box.row(); hint.enabled = False
    hint.label(
        text="The key is stored in your operating-system credential vault.",
        icon='LOCKED',
    )
