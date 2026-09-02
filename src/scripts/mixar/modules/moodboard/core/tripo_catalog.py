# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Catalog helpers shared by Tripo's Model Gen UI and operator."""


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
                for key in ("slug", "label", "name", "provider", "provider_id")
            )
            if "tripo" in text.lower():
                return True
    except Exception:
        pass
    return False
