# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compatibility wrapper for the former P1 multiview queue job."""

from .tripo_client import TRIPO_MODEL_P1
from .tripo_direct_job import TripoDirectGenerationJob


class TripoP1MultiViewJob(TripoDirectGenerationJob):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("input_mode", "MULTI")
        kwargs.setdefault("api_model", TRIPO_MODEL_P1)
        super().__init__(*args, **kwargs)


__all__ = ("TripoP1MultiViewJob",)
