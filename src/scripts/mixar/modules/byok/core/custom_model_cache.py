# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stable EnumProperty storage for models discovered from a custom endpoint."""

_items = [("__manual__", "Enter model manually", "Use the free-text model field")]


def get_items():
    return _items


def replace(model_ids):
    global _items
    clean = sorted({str(value).strip() for value in model_ids if str(value).strip()})
    _items = [("__manual__", "Enter model manually", "Use the free-text model field")]
    _items.extend((value, value, value) for value in clean)
