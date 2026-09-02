# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Catalog helpers shared by Tripo's Model Gen UI and operator."""


def _model_id(value):
    """Canonical identifier used only for exact Tripo model matching."""
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


# The catalog slug and Tripo's provider model id are two different
# vocabularies.  These are the versions for which this client implements the
# official /generation/multiview-to-model request.  Keep this list narrow: an
# unknown/legacy Tripo row must still fail closed rather than discard views.
_MULTI_VIEW_MODEL_IDS = frozenset(
    _model_id(value)
    for value in (
        "tripo-v31",
        "tripo-v3.1",
        "tripo-3.1",
        "v3.1-20260211",
        "tripo-p1",
        "P1-20260311",
    )
)


def tripo_supports_multi_view(service_key, model_slug):
    """True for Tripo versions whose multi-view wire contract is implemented.

    This is a client capability override for stale catalog rows.  The backend
    should also set ``_supports_multi_view`` on those rows, but older deployed
    catalogs can otherwise reject Tripo 3.1/P1 before their supported request
    path is reached.
    """
    candidates = [model_slug]
    try:
        from mixar.bootstrap.generation_catalog_cache import get_model

        record = get_model(service_key, model_slug)
        if isinstance(record, dict):
            candidates.extend(
                record.get(key)
                for key in (
                    "slug",
                    "provider_model",
                    "provider_model_id",
                    "model_id",
                    "model_version",
                    "version",
                )
            )
    except Exception:
        pass
    return any(_model_id(value) in _MULTI_VIEW_MODEL_IDS for value in candidates)


def is_tripo_generation_model(service_key, model_slug):
    """Recognize current and future catalog Tripo entries."""
    if "tripo" in str(model_slug or "").lower():
        return True
    try:
        from mixar.bootstrap.generation_catalog_cache import get_model, get_service

        records = (get_model(service_key, model_slug), get_service(service_key))
        for record in records:
            if not isinstance(record, dict):
                continue
            text = " ".join(
                str(record.get(key) or "")
                for key in (
                    "slug",
                    "label",
                    "name",
                    "provider",
                    "provider_id",
                    "provider_model",
                    "provider_model_id",
                )
            )
            if "tripo" in text.lower():
                return True
    except Exception:
        pass
    return False
