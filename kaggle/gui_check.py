# SPDX-FileCopyrightText: 2026 aimcrazyxx
# SPDX-License-Identifier: GPL-3.0-or-later

"""GUI start-up check for a freshly built Mixar, under a virtual display.

  MIXAR_GUI_CHECK_OUT=logs xvfb-run -a mixar --factory-startup --python gui_check.py

Waits for the bootstrap's deferred UI loading (it only runs from timers, so
the background-mode smoke test never reaches it), records every warning and
error Mixar logs on the way, checks that the paint layer stack registered,
saves a screenshot and quits. The verdict goes to gui_check.json.
"""

import json
import logging
import os
import sys
import time

import bpy

OUT_DIR = os.environ.get("MIXAR_GUI_CHECK_OUT", ".")
TIMEOUT_S = 300
_started = time.monotonic()
_records = []

_call_handlers = logging.Logger.callHandlers


def _record(self, record):
    if record.levelno >= logging.WARNING:
        _records.append({"level": record.levelname, "logger": record.name,
                         "message": record.getMessage()[:500]})
    return _call_handlers(self, record)


logging.Logger.callHandlers = _record


def _paint_registered():
    return bool(bpy.types.ShaderNodeTree.bl_rna.properties.get("mp")
                and getattr(bpy.types, "MLayer", None))


def _screenshot(window):
    path = os.path.abspath(os.path.join(OUT_DIR, "gui_check.png"))
    try:
        with bpy.context.temp_override(window=window, screen=window.screen):
            bpy.ops.screen.screenshot(filepath=path, check_existing=False)
        return path
    except Exception as exc:  # noqa: BLE001 - a screenshot is a bonus
        print(f"GUI screenshot failed: {exc}")
        return ""


def _finish():
    boot = sys.modules.get("bootstrap")
    complete = bool(boot and getattr(boot, "_ui_loading_complete", False))
    ui_errors = [r for r in _records if r["level"] in ("ERROR", "CRITICAL")]
    result = {
        "ui_loading_complete": complete,
        "ui_modules_registered": len(getattr(boot, "_loaded_ui_modules", ()) or ()),
        "paint_layer_stack_registered": _paint_registered(),
        "seconds": round(time.monotonic() - _started, 1),
        "errors": ui_errors,
        "warnings": [r for r in _records if r["level"] == "WARNING"],
    }
    result["ok"] = complete and result["paint_layer_stack_registered"] and not ui_errors
    window = bpy.context.window_manager.windows[0]
    result["screenshot"] = _screenshot(window)
    with open(os.path.join(OUT_DIR, "gui_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"GUI ok={result['ok']} ui_complete={complete} "
          f"ui_modules={result['ui_modules_registered']} "
          f"paint={result['paint_layer_stack_registered']} errors={len(ui_errors)} "
          f"warnings={len(result['warnings'])} after {result['seconds']}s")
    for record in ui_errors[:30]:
        print(f"GUI ERROR {record['logger']}: {record['message']}")
    with bpy.context.temp_override(window=window):
        bpy.ops.wm.quit_blender()


def _tick():
    boot = sys.modules.get("bootstrap")
    if getattr(boot, "_ui_loading_complete", False) or time.monotonic() - _started > TIMEOUT_S:
        # One more second lets the last batch's redraw settle before the shot.
        bpy.app.timers.register(_finish, first_interval=1.0)
        return None
    return 0.5


bpy.app.timers.register(_tick, first_interval=1.0)
