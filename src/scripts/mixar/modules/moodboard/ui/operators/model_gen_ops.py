# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model Gen Operators

Generate operator for the consolidated, catalog-driven Model Gen tab.
Routes by the tab's selected mode (catalog service of capability
``model_gen``) and submits through the unified job queue with each mode's
EXISTING wire payload shape (see generation_params/core/assemblers.py):

- ``model_3d``      → job_type "model_3d", FEATURE_MODEL_3D
- ``image_to_3d``   → job_type "image_to_3d" (Hunyuan Pro),
                      FEATURE_IMAGE_TO_3D_PRO, reuses the Pro import hook
- ``hunyuan_rapid`` → job_type "hunyuan_rapid", FEATURE_HUNYUAN_RAPID
"""

import base64 as _b64

from bpy.types import Operator
from mixar.config.logging_config import get_logger
from mixar.modules.moodboard.core.media_utils import is_still_item
from mixar.modules.moodboard.core.tripo_catalog import is_tripo_generation_model

logger = get_logger(__name__)

# Placeholder enum ids that are never real catalog keys/slugs.
_PLACEHOLDERS = ("LOADING", "ERROR", "NONE", "")


def _routing(service_key):
    """Per-service enqueue kwargs (feature key, flags, import hooks).

    Mirrors what each mode's legacy operator passes to
    ``enqueue_generation`` today. Unknown (future) services fall back to
    the catalog service's own feature_key and the model_3d-style listener.
    """
    from mixar.modules.common.job_queue.constants import (
        FEATURE_HUNYUAN_RAPID,
        FEATURE_IMAGE_TO_3D_PRO,
        FEATURE_MODEL_3D,
    )

    if service_key == "model_3d":
        return dict(
            feature_key=FEATURE_MODEL_3D,
            fail_message="3D model generation failed",
            scene_flag="mixie_image_to_3d_is_generating",
            batch_popup_title="Image to 3D batch complete",
        )
    if service_key == "image_to_3d":
        # on_imported is set by the operator (mesh naming + normalization),
        # so it is intentionally omitted here.
        return dict(
            feature_key=FEATURE_IMAGE_TO_3D_PRO,
            fail_message="Image to 3D failed",
            scene_flag="mixie_image_to_3d_is_generating",
            batch_popup_title="Image to 3D batch complete",
        )
    if service_key == "hunyuan_rapid":
        return dict(
            feature_key=FEATURE_HUNYUAN_RAPID,
            scene_flag="mixie_hunyuan_rapid_is_generating",
        )

    feature_key = service_key
    try:
        from mixar.bootstrap.generation_catalog_cache import get_service
        svc = get_service(service_key)
        if svc and svc.get("feature_key"):
            feature_key = svc["feature_key"]
    except Exception as exc:
        logger.debug(
            "Could not resolve catalog feature for %s: %s",
            service_key,
            type(exc).__name__,
        )
    return dict(
        feature_key=feature_key,
        scene_flag="mixie_image_to_3d_is_generating",
    )


class MIXIE_OT_model_gen_generate(Operator):
    """Generate a 3D model using the selected Model Gen mode"""

    bl_idname = "mixie.model_gen_generate"
    bl_label = "Generate 3D Model"
    bl_description = "Generate a 3D model using the selected mode and model"
    bl_options = {"REGISTER"}

    def _get_input_image(self, context, tab):
        """Input image from the shared image-source UI (or None)."""
        scene = context.scene
        if getattr(tab, 'use_selected_image', False):
            if hasattr(scene, 'mixie_moodboard_images'):
                for item in scene.mixie_moodboard_images:
                    if item.selected and is_still_item(item):
                        return item.image
            return None
        return getattr(tab, 'reference_image', None)

    def _turnaround_payload(self, context, image, service_key, model):
        """Multi-view payload fragment for the set *image* is the main of."""
        from mixar.modules.moodboard.core.turnaround_views import (
            build_active_group_payload,
        )

        if image is None:
            return None
        try:
            result = build_active_group_payload(
                context.scene, image, service_key, model)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return False
        if result is None:
            return None
        fragment, warnings = result
        for warning in warnings:
            self.report({"WARNING"}, warning)
        return fragment

    def execute(self, context):
        from mixar.modules.common.generation_params import (
            assemble_payload,
            collect_params,
            model_supports_multi_view,
            resolve_service_key,
        )
        from mixar.modules.common.utils.image_utils import (
            compress_for_service,
            compress_image_for_upload,
        )

        scene = context.scene
        sidebar = getattr(scene, 'mixie_moodboard_sidebar', None)
        tab = getattr(sidebar, 'tab_image_to_3d', None) if sidebar else None
        if tab is None:
            self.report({"WARNING"}, "Model Gen tab not available")
            return {"CANCELLED"}

        service_key = resolve_service_key("model_gen", getattr(tab, "mode", ""))
        if not service_key:
            self.report({"WARNING"}, "Please wait for the catalog to load")
            return {"CANCELLED"}

        model = getattr(tab, 'model', '')
        if model in _PLACEHOLDERS:
            try:
                from mixar.bootstrap.generation_catalog_cache import (
                    get_default_model_slug,
                )
                model = get_default_model_slug(service_key) or ""
            except Exception:
                model = ""
        if not model or model in _PLACEHOLDERS:
            self.report({"WARNING"}, "Please wait for models to load")
            return {"CANCELLED"}

        is_tripo = is_tripo_generation_model(service_key, model)
        if is_tripo and getattr(tab, "tripo_use_direct_api", False):
            return self._execute_tripo_direct(context, tab, model, service_key)

        if is_tripo and getattr(tab, 'tripo_input_mode', 'SINGLE') == 'MULTI':
            return self._execute_tripo_multi_backend(
                context, tab, model, service_key
            )

        image = self._get_input_image(context, tab)
        prompt = (getattr(tab, 'prompt', '') or '').strip() or None
        supports_mv = model_supports_multi_view(service_key, model)

        turnaround_payload = self._turnaround_payload(
            context, image, service_key, model)
        if turnaround_payload is False:
            return {"CANCELLED"}

        if turnaround_payload:
            pass
        elif service_key == "image_to_3d" or supports_mv:
            if not (image or prompt):
                self.report(
                    {"WARNING"},
                    "Provide at least one of: prompt, image, or multiple views",
                )
                return {"CANCELLED"}
        elif service_key == "hunyuan_rapid":
            if not (image or prompt):
                self.report({"WARNING"}, "Provide either a prompt or an image")
                return {"CANCELLED"}
        else:
            if not image:
                self.report({"WARNING"}, "Please add an input image")
                return {"CANCELLED"}

        payload = {}
        if turnaround_payload:
            payload.update(turnaround_payload)
        elif image is not None:
            try:
                if service_key == "model_3d":
                    image_bytes = compress_for_service(image, "image_to_3d")
                else:
                    image_bytes = compress_image_for_upload(image)
            except Exception as e:
                self.report({"ERROR"}, f"Failed to process image: {e}")
                return {"CANCELLED"}
            if image_bytes:
                payload["image_bytes_b64"] = _b64.b64encode(image_bytes).decode()
                payload["image_filename"] = "image.png"

        params = {}
        try:
            params = collect_params(service_key, model)
        except Exception as e:
            logger.debug(
                "collect_params failed for %s/%s: %s", service_key, model, e
            )
        if prompt:
            if service_key == "image_to_3d":
                params["prompt"] = prompt
            elif service_key == "hunyuan_rapid":
                if image is None:
                    params["prompt"] = prompt
            else:
                payload["prompt"] = prompt
        payload = assemble_payload(service_key, params, payload, model)

        route = _routing(service_key)
        feature_key = route.pop("feature_key")
        label = image.name if image else ((prompt or model)[:40])

        # Name the imported mesh from the input image (or a prompt slug for
        # text-to-3D) and normalize its placement. Overrides any service
        # default hook (e.g. image_to_3d's legacy "_high" rename) so every
        # Model Gen provider names + grounds its result consistently.
        from mixar.modules.moodboard.core.generation_enqueue import (
            derive_model_name, make_model_rename_on_imported, model_front_zrot,
        )
        mesh_name = derive_model_name(image, prompt or "")
        route["on_imported"] = make_model_rename_on_imported(
            mesh_name, model_front_zrot(model))

        try:
            from mixar.modules.common.job_queue import enqueue_generation

            job = enqueue_generation(
                kind="glb",
                feature_key=feature_key,
                job_type=service_key,
                model=model,
                payload=payload,
                label=label,
                **route,
            )
            if not job:
                self.report({"WARNING"}, "A duplicate generation is already queued")
                return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to start generation: {e}")
            return {"CANCELLED"}

        from mixar.modules.common.job_queue.ui.lists.queue_uilist import (
            mark_enqueued,
        )
        mark_enqueued(feature_key)
        self.report({"INFO"}, "Added to queue")
        return {"FINISHED"}

    def _execute_tripo_direct(self, context, tab, model, service_key):
        """Submit text, image, or multiview generation to Tripo v3 BYOK."""
        from mixar.modules.common.secure_storage import (
            get_secret,
            masked_preview,
            set_secret,
        )
        from mixar.modules.common.utils.image_utils import compress_for_service
        from mixar.modules.moodboard.core.generation_enqueue import (
            derive_model_name,
            make_model_rename_on_imported,
            model_front_zrot,
        )
        from mixar.modules.moodboard.core.tripo_client import validate_api_key
        from mixar.modules.moodboard.core.tripo_direct_job import (
            enqueue_tripo_direct_job,
        )

        entered_key = (getattr(tab, "tripo_api_key", "") or "").strip()
        try:
            api_key = validate_api_key(entered_key or get_secret("tripo_api_key"))
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if entered_key:
            if not set_secret("tripo_api_key", api_key):
                self.report({"ERROR"}, "Could not store the Tripo API key securely")
                return {"CANCELLED"}
            tab.tripo_key_preview = masked_preview(api_key)
            tab.tripo_api_key = ""

        prompt = (getattr(tab, "prompt", "") or "").strip()
        selected_mode = getattr(tab, "tripo_input_mode", "SINGLE")
        image_refs = {}
        if selected_mode == "MULTI":
            image_refs = {
                "front": getattr(tab, "tripo_front_image", None),
                "left": getattr(tab, "tripo_left_image", None),
                "back": getattr(tab, "tripo_back_image", None),
                "right": getattr(tab, "tripo_right_image", None),
            }
            present = {key: value for key, value in image_refs.items() if value}
            if image_refs["front"] is None or len(present) < 2:
                self.report(
                    {"ERROR"},
                    "Multi View requires Front plus at least one other view",
                )
                return {"CANCELLED"}
            input_mode = "MULTI"
            if prompt:
                self.report(
                    {"WARNING"},
                    "Direct Tripo Multi View uses the views; prompt is ignored",
                )
        else:
            image = self._get_input_image(context, tab)
            if image is not None:
                image_refs = {"front": image}
                input_mode = "SINGLE"
                if prompt:
                    self.report(
                        {"WARNING"},
                        "Direct Tripo image mode uses the image; prompt is ignored",
                    )
            elif prompt:
                input_mode = "TEXT"
            else:
                self.report({"ERROR"}, "Provide an image or a prompt")
                return {"CANCELLED"}

        images = {}
        try:
            for view, image in image_refs.items():
                if image is None:
                    continue
                data = compress_for_service(image, "image_to_3d")
                if not data:
                    raise ValueError(f"'{image.name}' has no pixel data")
                images[view] = data
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to process Tripo image: {exc}")
            return {"CANCELLED"}

        route = _routing(service_key)
        feature_key = route["feature_key"]
        front_image = image_refs.get("front")
        label = (
            f"{front_image.name} ({'Multi View' if input_mode == 'MULTI' else 'Tripo'})"
            if front_image
            else (prompt[:40] or "Tripo")
        )
        mesh_name = derive_model_name(front_image, prompt)
        on_imported = make_model_rename_on_imported(
            mesh_name, model_front_zrot(model)
        )

        try:
            job = enqueue_tripo_direct_job(
                feature_key=feature_key,
                label=label,
                scene_flag=route.get("scene_flag", ""),
                batch_popup_title=route.get("batch_popup_title", ""),
                service=service_key,
                origin_capability_key="model_gen",
                model=model,
                input_mode=input_mode,
                api_model=getattr(tab, "tripo_api_model", "v3.1-20260211"),
                images=images,
                prompt=prompt,
                api_key=api_key,
                texture=getattr(tab, "tripo_texture", True),
                pbr=getattr(tab, "tripo_pbr", True),
                face_limit=getattr(tab, "tripo_face_limit", 0),
                model_seed=getattr(tab, "tripo_model_seed", 0),
                texture_quality=getattr(
                    tab, "tripo_texture_quality", "standard"
                ),
                geometry_quality=getattr(
                    tab, "tripo_geometry_quality", "standard"
                ),
                texture_alignment=getattr(
                    tab, "tripo_texture_alignment", "original_image"
                ),
                orientation=getattr(tab, "tripo_orientation", "default"),
                _on_imported_hook=on_imported,
            )
            if not job:
                self.report({"WARNING"}, "A duplicate generation is already queued")
                return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to start direct Tripo generation: {exc}")
            return {"CANCELLED"}

        from mixar.modules.common.job_queue.ui.lists.queue_uilist import mark_enqueued

        mark_enqueued(feature_key)
        self.report({"INFO"}, "Direct Tripo generation added to queue")
        return {"FINISHED"}

    def _execute_tripo_multi_backend(self, context, tab, model, service_key):
        """Submit Tripo Multi View through the same Mixar backend as Single.

        The front image uses the normal top-level image contract and the other
        views use the existing generic ``multi_view_images`` contract. This
        deliberately does not read or store a Tripo API key: authentication,
        credits, provider routing, polling, download and import are all handled
        by the ordinary Mixar ``model_3d`` job queue path.
        """
        from mixar.modules.common.generation_params import (
            assemble_payload,
            collect_params,
        )
        from mixar.modules.common.utils.image_utils import compress_for_service

        images = {
            "front": getattr(tab, "tripo_front_image", None),
            "left": getattr(tab, "tripo_left_image", None),
            "back": getattr(tab, "tripo_back_image", None),
            "right": getattr(tab, "tripo_right_image", None),
        }
        present = {name: image for name, image in images.items() if image is not None}
        if images["front"] is None or len(present) < 2:
            self.report(
                {"ERROR"},
                "Multi View requires Front plus at least one other view",
            )
            return {"CANCELLED"}

        encoded = {}
        try:
            for view, image in present.items():
                data = compress_for_service(image, "image_to_3d")
                if not data:
                    raise ValueError(f"'{image.name}' has no pixel data")
                encoded[view] = _b64.b64encode(data).decode()
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to process Multi View image: {exc}")
            return {"CANCELLED"}

        payload = {
            "image_bytes_b64": encoded["front"],
            "image_filename": "front.png",
            "multi_view_images": [
                {
                    "image_bytes_b64": encoded[view],
                    "filename": f"{view}.png",
                    "view_type": view,
                }
                for view in ("left", "back", "right")
                if view in encoded
            ],
        }

        prompt = (getattr(tab, "prompt", "") or "").strip() or None
        params = {}
        try:
            params = collect_params(service_key, model)
        except Exception as exc:
            logger.debug(
                "collect_params failed for %s/%s: %s",
                service_key,
                model,
                exc,
            )
        if prompt:
            payload["prompt"] = prompt
        payload = assemble_payload(service_key, params, payload, model)

        route = _routing(service_key)
        feature_key = route.pop("feature_key")
        label = f"{images['front'].name} (Multi View)"

        try:
            from mixar.modules.common.job_queue import enqueue_generation

            job = enqueue_generation(
                kind="glb",
                feature_key=feature_key,
                job_type=service_key,
                model=model,
                payload=payload,
                label=label,
                **route,
            )
            if not job:
                self.report({"WARNING"}, "A duplicate generation is already queued")
                return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to start Multi View generation: {exc}")
            return {"CANCELLED"}

        from mixar.modules.common.job_queue.ui.lists.queue_uilist import mark_enqueued

        mark_enqueued(feature_key)
        self.report({"INFO"}, "Multi View added to Mixar queue")
        return {"FINISHED"}


class MIXIE_OT_tripo_clear_api_key(Operator):
    """Remove the direct Tripo credential from the OS credential vault."""

    bl_idname = "mixie.tripo_clear_api_key"
    bl_label = "Remove Saved Tripo Key"
    bl_description = "Remove the saved Tripo API key from this computer"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from mixar.modules.common.secure_storage import delete_secret

        if not delete_secret("tripo_api_key"):
            self.report({"ERROR"}, "Could not remove the saved Tripo API key")
            return {"CANCELLED"}
        sidebar = getattr(context.scene, "mixie_moodboard_sidebar", None)
        tab = getattr(sidebar, "tab_image_to_3d", None) if sidebar else None
        if tab is not None:
            tab.tripo_api_key = ""
            tab.tripo_key_preview = ""
        self.report({"INFO"}, "Saved Tripo API key removed")
        return {"FINISHED"}


classes = (
    MIXIE_OT_model_gen_generate,
    MIXIE_OT_tripo_clear_api_key,
)


def register():
    """Register operator classes"""
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)


def unregister():
    """Unregister operator classes"""
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
