# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model Gen Tab Drawer — consolidated, catalog-driven 3D generation.

Renders the Model Gen sidebar tab from the generation catalog's
``model_gen`` capability: a Mode dropdown (Image to 3D / Image to 3D Pro /
Rapid 3D), the selected mode's Model dropdown, its schema-driven params,
and the shared input-image UI. Called from
``sidebar_panel_drawers._draw_image_to_3d`` which falls back to the legacy
Basic/Pro subtab UI when the catalog isn't loaded.
"""

from .sidebar_ui_helpers import (
    draw_generate_footer,
    draw_image_info_card,
    draw_image_thumbnail,
    draw_moodboard_image_toggle,
    draw_prompt_section,
    draw_section_box,
    draw_section_separator,
)
from ..core.tripo_catalog import is_tripo_generation_model

_MODEL_GEN_GENERATE = "mixie.model_gen_generate"
_MODEL_GEN_FOOTER = {
    "model_3d": ("model_3d", "mixie_image_to_3d_is_generating",
                 _MODEL_GEN_GENERATE),
    "image_to_3d": ("image_to_3d_pro", "mixie_image_to_3d_is_generating",
                    _MODEL_GEN_GENERATE),
    "hunyuan_rapid": ("hunyuan_rapid", "mixie_hunyuan_rapid_is_generating",
                      _MODEL_GEN_GENERATE),
    "tripo_smart_segment": ("smart_segment", "mixie_smart_segment_is_generating",
                            "mixie.smart_segment_generate"),
}


def _model_gen_catalog_ready():
    """True when the catalog has model_gen services (drives the new UI)."""
    try:
        from mixar.bootstrap.generation_catalog_cache import (
            get_services,
            is_loaded,
        )
        return is_loaded() and bool(get_services("model_gen"))
    except Exception:
        return False


def _draw_model_gen(layout, context):
    """Draw the consolidated, catalog-driven Model Gen tab."""
    scene = context.scene
    tab = scene.mixie_moodboard_sidebar.tab_image_to_3d

    from mixar.modules.common.generation_params import (
        draw_capability_selector,
        resolve_service_key,
    )
    service_key = resolve_service_key(
        "model_gen", getattr(tab, "mode", "")
    ) or "model_3d"
    model_slug = getattr(tab, "model", "")
    is_tripo = is_tripo_generation_model(service_key, model_slug)

    draw_prompt_section(layout, tab, label="Prompt (optional)")
    draw_section_separator(layout)

    if is_tripo:
        tabs = layout.row(align=True)
        tabs.prop(tab, "tripo_input_mode", expand=True)

    is_tripo_multi = is_tripo and tab.tripo_input_mode == 'MULTI'
    if is_tripo_multi:
        col = draw_section_box(layout, "Multi View Images", icon='RENDERLAYERS')
        grid = col.grid_flow(row_major=True, columns=2, even_columns=True)
        for prop, label in (
            ('tripo_front_image', "Front (required)"),
            ('tripo_left_image', "Left"),
            ('tripo_back_image', "Back"),
            ('tripo_right_image', "Right"),
        ):
            cell = grid.column()
            cell.label(text=label)
            image = getattr(tab, prop, None)
            if image is not None:
                draw_image_thumbnail(cell, image, scale=2.2)
            cell.template_ID(tab, prop, open="image.open")
        col.label(
            text="Front plus at least one other view is required.",
            icon='INFO',
        )
        if not tab.tripo_use_direct_api:
            col.label(text="Uses the same Mixar backend as Single.", icon='CHECKMARK')
    else:
        col = draw_section_box(
            layout,
            "Input Image",
            icon='IMAGE_DATA',
            action_op="mixie.image_to_3d_pick_image",
        )
        draw_moodboard_image_toggle(col, tab, context)
        if not tab.use_selected_image and tab.reference_image:
            draw_image_info_card(
                col, tab.reference_image,
                remove_op="mixie.image_to_3d_remove_image",
            )

    from .turnaround_drawer import draw_detect_views_section
    draw_section_separator(layout)
    if not is_tripo_multi:
        draw_detect_views_section(layout, context, service_key, model_slug)

    draw_section_separator(layout)

    col = draw_section_box(layout, "Settings", icon='SETTINGS')
    col.use_property_split = True
    col.use_property_decorate = False
    draw_capability_selector(
        col, tab, "model_gen",
        model_refresh_op="mixie.image_to_3d_refresh_models",
    )

    if is_tripo:
        col.separator()
        col.prop(tab, "tripo_use_direct_api")
        if tab.tripo_use_direct_api:
            col.prop(tab, "tripo_api_model")
            col.prop(tab, "tripo_api_key")
            if tab.tripo_key_preview:
                col.label(
                    text=f"Saved securely: {tab.tripo_key_preview}",
                    icon='LOCKED',
                )
            else:
                col.label(
                    text="Enter tsk_ key, or leave blank to use a saved key",
                    icon='INFO',
                )
            col.operator(
                "mixie.tripo_clear_api_key",
                text="Remove Saved Key",
                icon='TRASH',
            )
            col.prop(tab, "tripo_texture")
            texture_row = col.row()
            texture_row.enabled = tab.tripo_texture
            texture_row.prop(tab, "tripo_pbr")
            col.prop(tab, "tripo_face_limit")
            col.prop(tab, "tripo_geometry_quality")
            texture_quality_row = col.row()
            texture_quality_row.enabled = tab.tripo_texture
            texture_quality_row.prop(tab, "tripo_texture_quality")
            col.prop(tab, "tripo_texture_alignment")
            col.prop(tab, "tripo_orientation")
            col.prop(tab, "tripo_model_seed")
            warning = col.column(align=True)
            warning.alert = True
            warning.label(text="Uses your Tripo credits directly.", icon='ERROR')
            warning.label(text="Local cancel may not stop a remote task.")

    feature_key, gen_flag, generate_op = _MODEL_GEN_FOOTER.get(
        service_key, _MODEL_GEN_FOOTER["model_3d"])
    draw_generate_footer(
        layout, context, generate_op, "image_to_3d",
        gen_flag_attr=gen_flag, feature_key=feature_key,
    )
