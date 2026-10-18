#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 aimcrazyxx
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Push kaggle/mixar-linux-build.ipynb to your Kaggle account and start a run.
#
#   kaggle/push_kernel.sh [--ref BRANCH] [--mixar-ref REF] [--slug NAME] [--wait] [--download DIR]
#
#   --ref        branch/tag of THIS repository the notebook clones (default: main)
#   --mixar-ref  official Mixar ref to build (default: latest-release = newest release)
#   --slug       Kaggle notebook to push to (default: KAGGLE_KERNEL_SLUG or mixar-linux-build)
#   --wait       poll until the Kaggle run finishes; exit non-zero if it failed
#   --download   after --wait, fetch the run's output (tarball, logs) into DIR
#   --force      push even while an earlier version of the notebook still runs
#
# Kaggle cancels a version pushed while an earlier version of the same notebook
# is still running ("Canceled by backend", a few minutes in), so the script
# refuses to push then: wait for that run, or build in another notebook (--slug).
#
# Needs the Kaggle CLI (pip install kaggle) and an API token: KAGGLE_API_TOKEN,
# or ~/.kaggle/access_token. KAGGLE_USERNAME is looked up when unset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP3_REF="main"
MIXAR_REF="latest-release"
WAIT=0
FORCE=0
DOWNLOAD=""
SLUG="${KAGGLE_KERNEL_SLUG:-mixar-linux-build}"

while (( $# )); do
    case "$1" in
        --ref) APP3_REF="$2"; shift 2 ;;
        --mixar-ref) MIXAR_REF="$2"; shift 2 ;;
        --slug) SLUG="$2"; shift 2 ;;
        --wait) WAIT=1; shift ;;
        --download) DOWNLOAD="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '5,21p' "$0"; exit 0 ;;
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
# Kaggle derives a notebook's slug from its title, so the title must match --slug.
meta["title"] = " ".join(word.capitalize() for word in kernel.split("/")[1].split("-"))
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

# Versions of the notebook still queued or running. Best effort: when the API
# cannot tell (a new notebook, a transient error), the push goes ahead.
running="$(python3 - "$USER_NAME" "$SLUG" 2>/dev/null <<'PY' | grep -E '^v[0-9]+$' || true
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionStatusRequest

user, slug = sys.argv[1:]
api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    client = kaggle.kernels.kernels_api_client
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = user, slug
    latest = client.get_kernel(request).metadata.current_version_number or 0
    for version in range(1, latest + 1):
        request = ApiGetKernelSessionStatusRequest()
        request.user_name, request.kernel_slug, request.version_label = user, slug, f"v{version}"
        try:
            status = client.get_kernel_session_status(request).status.name
        except Exception:
            continue
        if status in ("QUEUED", "RUNNING"):
            print(f"v{version}")
PY
)"
if [[ -n "$running" ]] && (( ! FORCE )); then
    echo "$KERNEL is still running $(echo $running), and Kaggle would cancel a version pushed" \
         "now. Wait for that run to finish (or cancel it on kaggle.com), build in another" \
         "notebook with --slug, or push anyway with --force." >&2
    exit 1
fi

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
