# SPDX-FileCopyrightText: 2026 aimcrazyxx
# SPDX-License-Identifier: GPL-3.0-or-later

"""Headless start-up check for a freshly built Mixar.

Run inside the app:  mixar -b --factory-startup --python-exit-code 3 --python smoke_test.py

Background mode never ticks the bootstrap's UI auto-discovery timers, so this
imports every Mixar module itself. An import error there is a feature that
would silently be missing from the GUI. Raises (exit code 3) when anything
fails to import.
"""

import importlib
import os
import sys
import traceback
from collections import Counter

import bpy

SKIP_DIRS = {"__pycache__", "tests", "testing", "vendor"}
BUNDLED = ("keyring", "truststore", "websocket", "httpx", "mcp", "mistune", "PIL", "cryptography")


def scripts_dir():
    path = bpy.utils.system_resource('SCRIPTS')
    if path not in sys.path:
        sys.path.append(path)
    return path


def module_names(root):
    base = os.path.dirname(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for filename in sorted(filenames):
            if not filename.endswith(".py") or filename.startswith("test_"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, filename), base)[:-3]
            parts = rel.split(os.sep)
            if parts[-1] == "__init__":
                parts.pop()
            yield ".".join(parts)


def main():
    print(f"SMOKE Blender {bpy.app.version_string}, Python {sys.version.split()[0]}")
    mixar_root = os.path.join(scripts_dir(), "mixar")
    if not os.path.isdir(mixar_root):
        raise RuntimeError(f"Mixar scripts are missing from {mixar_root}")

    failures = []
    names = list(module_names(mixar_root))
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    print(f"SMOKE modules: {len(names) - len(failures)}/{len(names)} imported")
    for name, error in failures[:60]:
        print(f"  FAIL {name}: {error}")
    for error, count in Counter(error for _name, error in failures).most_common(10):
        print(f"SMOKE cause x{count}: {error}")

    missing = []
    for package in BUNDLED:
        try:
            importlib.import_module(package)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{package} ({type(exc).__name__}: {exc})")
    print("SMOKE bundled packages:", "all importable" if not missing else "MISSING " + ", ".join(missing))

    if failures or missing:
        raise RuntimeError(f"{len(failures)} module(s) failed to import, {len(missing)} package(s) missing")
    print("SMOKE OK")


main()
