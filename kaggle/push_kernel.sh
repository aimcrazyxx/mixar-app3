#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 aimcrazyxx
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Push kaggle/mixar-linux-build.ipynb to your Kaggle account and start a run.
#
#   kaggle/push_kernel.sh [--ref BRANCH] [--mixar-ref REF] [--wait] [--download DIR]
#
#   --ref        branch/tag of THIS repository the notebook clones (default: main)
#   --mixar-ref  official Mixar ref to build (default: main = newest code)
#   --wait       poll until the Kaggle run finishes; exit non-zero if it failed
#   --download   after --wait, fetch the run's output (tarball, logs) into DIR
#
# Needs the Kaggle CLI (pip install kaggle) and an API token: KAGGLE_API_TOKEN,
# or ~/.kaggle/access_token. KAGGLE_USERNAME is looked up when unset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP3_REF="main"
MIXAR_REF="main"
WAIT=0
DOWNLOAD=""
SLUG="${KAGGLE_KERNEL_SLUG:-mixar-linux-build}"

while (( $# )); do
    case "$1" in
        --ref) APP3_REF="$2"; shift 2 ;;
        --mixar-ref) MIXAR_REF="$2"; shift 2 ;;
        --wait) WAIT=1; shift ;;
        --download) DOWNLOAD="$2"; shift 2 ;;
        -h|--help) sed -n '5,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v kaggle >/dev/null 2>&1 || { echo "install the Kaggle CLI: pip install kaggle" >&2; exit 1; }
USER_NAME="${KAGGLE_USERNAME:-$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')}"
[[ -n "$USER_NAME" ]] || { echo "could not determine your Kaggle username; set KAGGLE_USERNAME" >&2; exit 1; }
KERNEL="$USER_NAME/$SLUG"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

python3 - "$SCRIPT_DIR" "$STAGE" "$KERNEL" "$APP3_REF" "$MIXAR_REF" <<'PY'
import json
import re
import sys

src, dst, kernel, app3_ref, mixar_ref = sys.argv[1:]
with open(f"{src}/kernel-metadata.json") as f:
    meta = json.load(f)
meta["id"] = kernel
with open(f"{dst}/kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

with open(f"{src}/{meta['code_file']}") as f:
    nb = json.load(f)
values = {"MIXAR_APP3_REF": app3_ref, "MIXAR_REF": mixar_ref}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    lines = []
    for line in cell["source"]:
        for name, value in values.items():
            line = re.sub(rf'^{name} = "[^"]*"', f"{name} = {json.dumps(value)}", line)
        lines.append(line)
    cell["source"] = lines
with open(f"{dst}/{meta['code_file']}", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
PY

echo "Pushing $KERNEL (repo ref $APP3_REF, Mixar ref $MIXAR_REF)"
kaggle kernels push -p "$STAGE"
echo "Follow it at https://www.kaggle.com/code/$KERNEL"

(( WAIT )) || exit 0

last=""
while true; do
    status="$(kaggle kernels status "$KERNEL" 2>&1 | tail -n 1 || true)"
    if [[ "$status" != "$last" ]]; then
        echo "[$(date -u +%H:%M:%S)] $status"
        last="$status"
    fi
    # Only the CLI's "<kernel> has status ..." line is a verdict; anything else
    # (a transient API error) is retried on the next poll.
    case "${status,,}" in
        *"has status"*complete*) break ;;
        *"has status"*error*|*"has status"*cancel*)
            echo "Kaggle run did not complete" >&2
            if [[ -n "$DOWNLOAD" ]]; then
                mkdir -p "$DOWNLOAD"
                kaggle kernels output "$KERNEL" -p "$DOWNLOAD" || true
            fi
            exit 1 ;;
    esac
    sleep 120
done

if [[ -n "$DOWNLOAD" ]]; then
    mkdir -p "$DOWNLOAD"
    kaggle kernels output "$KERNEL" -p "$DOWNLOAD"
    ls -lh "$DOWNLOAD"
fi
