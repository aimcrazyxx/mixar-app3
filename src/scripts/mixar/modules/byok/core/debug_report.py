# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Secret-safe AI debug report serialization."""

import json
import re

_SECRET_KEYS = re.compile(r"(api.?key|authorization|token|secret|password|cookie)", re.I)


def redact(value):
    if isinstance(value, dict):
        return {
            str(key): ("[REDACTED]" if _SECRET_KEYS.search(str(key)) else redact(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?i)(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+", r"\1…[REDACTED]", value)
    return value


def to_json(report: dict) -> str:
    return json.dumps(redact(report), indent=2, ensure_ascii=False, sort_keys=True)
