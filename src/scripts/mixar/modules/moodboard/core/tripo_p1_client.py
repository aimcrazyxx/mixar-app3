# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compatibility names for the former P1-only Tripo client."""

from .tripo_client import (
    TRIPO_BASE_URL,
    TRIPO_MODEL_P1 as TRIPO_P1_MODEL,
    TripoClient,
    TripoError as TripoP1Error,
    _image_upload_metadata,
)


class TripoP1Client(TripoClient):
    def __init__(self, api_key: str, **kwargs):
        kwargs.setdefault("model", TRIPO_P1_MODEL)
        super().__init__(api_key, **kwargs)


__all__ = (
    "TRIPO_BASE_URL",
    "TRIPO_P1_MODEL",
    "TripoP1Client",
    "TripoP1Error",
    "_image_upload_metadata",
)
