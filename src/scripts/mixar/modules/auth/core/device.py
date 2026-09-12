# SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Random device identifier generated per session.

Sent to the backend at login, SSO token exchange, and the agent WebSocket
handshake as a device signal. A fresh random id is generated every time
the id is requested, so each login / session presents a different device
id to the backend (allowing multiple accounts on the same machine).
The raw OS machine identifier is never read, sent, or persisted.

All failure paths return None: the device id is best-effort and must never
break login or connection flows.
"""

import re
import uuid

from ....config.logging_config import get_logger

logger = get_logger(__name__)

_DEVICE_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")


def get_device_id():
    """Return a randomly generated device id.

    A fresh random id is generated every time this function is called,
    so each login / SSO token exchange / agent handshake session presents
    a different device id to the backend. No hardware identifier is read
    and nothing is persisted across sessions.
    """
    random_id = uuid.uuid4().hex
    if _DEVICE_ID_RE.match(random_id):
        return random_id
    # Fallback to a plain random hex string in the unlikely validation mismatch.
    return uuid.uuid4().hex
