# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small provider-neutral wrapper around the OS credential store.

Secrets handled here never enter Blender ID properties and therefore cannot be
written into a ``.blend``.  Windows reuses Mixar's native Credential Manager
adapter; macOS/Linux use the already bundled ``keyring`` backend.
"""

import platform

from mixar.config.logging_config import get_logger

logger = get_logger(__name__)

_SERVICE = "MixarBYOK"
_WINDOWS = platform.system() == "Windows"


def _safe_name(name: str) -> str:
    value = str(name or "").strip().lower()
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-." for ch in value):
        raise ValueError("Invalid secure-storage key name")
    return value


def get_secret(name: str) -> str:
    key = _safe_name(name)
    try:
        if _WINDOWS:
            from mixar.modules.auth.core import auth
            return auth._win_get_password(f"{key}@{_SERVICE}") or ""
        import keyring
        return keyring.get_password(_SERVICE, key) or ""
    except Exception as exc:
        logger.warning("Could not read %s credential: %s", key, type(exc).__name__)
        return ""


def set_secret(name: str, value: str) -> bool:
    key = _safe_name(name)
    secret = str(value or "")
    if not secret:
        return delete_secret(key)
    try:
        if _WINDOWS:
            from mixar.modules.auth.core import auth
            return bool(auth._win_set_password(f"{key}@{_SERVICE}", key, secret))
        import keyring
        keyring.set_password(_SERVICE, key, secret)
        return True
    except Exception as exc:
        logger.error("Could not store %s credential: %s", key, type(exc).__name__)
        return False


def delete_secret(name: str) -> bool:
    key = _safe_name(name)
    try:
        if _WINDOWS:
            from mixar.modules.auth.core import auth
            return bool(auth._win_delete_password(f"{key}@{_SERVICE}"))
        import keyring
        if keyring.get_password(_SERVICE, key) is not None:
            keyring.delete_password(_SERVICE, key)
        return True
    except Exception as exc:
        logger.warning("Could not delete %s credential: %s", key, type(exc).__name__)
        return False


def masked_preview(secret: str) -> str:
    value = str(secret or "")
    if not value:
        return ""
    suffix = value[-4:] if len(value) >= 4 else ""
    return f"••••••••{suffix}"
