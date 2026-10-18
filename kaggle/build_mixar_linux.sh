#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 aimcrazyxx
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build the newest Mixar (github.com/Mixar-AI/mixar-app) for Linux x86_64 and
# package it as a portable tarball.
#
# Made for Kaggle notebooks (Ubuntu 24.04, 4 vCPU, ~30 GB RAM, internet on),
# but runs on any Ubuntu 24.04 host with root or sudo.
#
# Every knob is an environment variable:
#   MIXAR_REF     official Mixar ref to build: main (default, newest code),
#                 latest-tag (newest vX.Y.Z tag) or any tag/branch/commit
#   MIXAR_REPO    official repository URL
#   MIXAR_ENV     Prod (default) | Dev | UAT
#   MIXAR_CUDA    0 (default, Cycles renders on CPU) | 1 (needs a CUDA toolkit)
#   PATCH_SERIES  file listing the patches to apply, one per line
#   WORK_DIR      scratch tree, ~15 GB (default /tmp/mixar-build)
#   OUT_DIR       tarball + logs (default /kaggle/working, else $WORK_DIR/out)
#   BUILD_CORES   parallel compile jobs (default: nproc)
#   SKIP_DEPS=1   do not apt-get install the toolchain
#   USE_CCACHE    1 compiles through ccache (default: 1 outside Kaggle)
#   SMOKE_TEST=0  skip the headless start-up check of the built app

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ON_KAGGLE=0
[[ -d /kaggle/working ]] && ON_KAGGLE=1

MIXAR_REPO="${MIXAR_REPO:-https://github.com/Mixar-AI/mixar-app.git}"
MIXAR_REF="${MIXAR_REF:-main}"
export MIXAR_ENV="${MIXAR_ENV:-Prod}"
export MIXAR_CUDA="${MIXAR_CUDA:-0}"
PATCH_SERIES="${PATCH_SERIES:-$REPO_ROOT/patches/series}"
WORK_DIR="${WORK_DIR:-/tmp/mixar-build}"
if [[ -z "${OUT_DIR:-}" ]]; then
    if (( ON_KAGGLE )); then OUT_DIR=/kaggle/working; else OUT_DIR="$WORK_DIR/out"; fi
fi
export BUILD_CORES="${BUILD_CORES:-$(nproc)}"
SKIP_DEPS="${SKIP_DEPS:-0}"
USE_CCACHE="${USE_CCACHE:-$(( ON_KAGGLE ? 0 : 1 ))}"
SMOKE_TEST="${SMOKE_TEST:-1}"

BLENDER_GIT="https://projects.blender.org/blender/blender.git"
LIBS_GIT="https://projects.blender.org/blender/lib-linux_x64.git"
MIXAR_DIR="$WORK_DIR/mixar"
LOG_DIR="$OUT_DIR/logs"
BUILD_LOG="$LOG_DIR/build.log"

mkdir -p "$WORK_DIR" "$LOG_DIR"
START_TS=$(date +%s)

step() {
    printf '\n==> [%s, +%d min] %s\n' "$(date -u +%H:%M:%S)" \
        $(( ($(date +%s) - START_TS) / 60 )) "$*"
}
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
trap 'rc=$?; printf "\nFAILED (exit %d) at line %d: %s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

SUDO=""
if [[ $EUID -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "run as root or install sudo"
    SUDO="sudo"
fi

# --------------------------------------------------------------------------
# Toolchain. Blender 5.2 needs GCC >= 14 (or Clang >= 17); Ubuntu 24.04's
# default gcc is 13, so gcc-14 is installed next to it and selected below.
# --------------------------------------------------------------------------
APT_PACKAGES=(
    build-essential gcc-14 g++-14 lld git git-lfs cmake ninja-build patch
    rsync xz-utils zstd ccache python3
    libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev
    libxinerama-dev libxkbcommon-dev libwayland-dev libdecor-0-dev
    wayland-protocols libdbus-1-dev libgl-dev libegl-dev
    libjack-jackd2-dev libpulse-dev libpipewire-0.3-dev
    # Mixar's native sign-in (source/creator): HTTPS via libcurl + OpenSSL,
    # tokens in the Secret Service keyring that Python's keyring also uses.
    pkg-config libcurl4-openssl-dev libssl-dev libsecret-1-dev
)

install_deps() {
    step "Installing the build toolchain"
    export DEBIAN_FRONTEND=noninteractive
    # Kaggle ships third-party apt lists that may warn; the Ubuntu ones suffice.
    $SUDO apt-get update -qq || echo "apt-get update reported errors; continuing"
    $SUDO apt-get install -y -qq --no-install-recommends "${APT_PACKAGES[@]}"
}

select_compiler() {
    if command -v gcc-14 >/dev/null 2>&1 && command -v g++-14 >/dev/null 2>&1; then
        export CC=gcc-14 CXX=g++-14
    else
        local cc_major
        cc_major="$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
        (( cc_major >= 14 )) || die "GCC 14 or newer is required (found gcc $cc_major); install gcc-14 g++-14"
        export CC=gcc CXX=g++
    fi
    echo "Compiler: $($CXX --version | head -n1)"
}

print_system() {
    step "System"
    grep PRETTY_NAME /etc/os-release || true
    echo "CPUs: $(nproc), jobs: $BUILD_CORES"
    free -g | sed -n 2p
    df -h "$WORK_DIR" "$OUT_DIR" | tail -n +2
}

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
resolve_mixar_ref() {
    if [[ "$MIXAR_REF" == "latest-tag" ]]; then
        MIXAR_REF="$(git ls-remote --tags --refs "$MIXAR_REPO" 'v*' \
            | sed 's#.*refs/tags/##' | grep -E '^v[0-9]+(\.[0-9]+)*$' | sort -V | tail -n1)"
        [[ -n "$MIXAR_REF" ]] || die "no release tag found in $MIXAR_REPO"
    fi
    echo "Mixar ref: $MIXAR_REF"
}

fetch_mixar() {
    step "Fetching Mixar ($MIXAR_REPO @ $MIXAR_REF)"
    if [[ ! -d "$MIXAR_DIR/.git" ]]; then
        git init -q "$MIXAR_DIR"
        git -C "$MIXAR_DIR" remote add origin "$MIXAR_REPO"
    fi
    git -C "$MIXAR_DIR" fetch -q --depth 1 origin "$MIXAR_REF"
    # Re-runs reuse the checkout: only files that differ get new mtimes, so
    # ninja recompiles what actually changed.
    git -C "$MIXAR_DIR" checkout -q -f --detach FETCH_HEAD
    git -C "$MIXAR_DIR" clean -q -fdx -e /upstream -e /build -e /source
    MIXAR_COMMIT="$(git -C "$MIXAR_DIR" rev-parse HEAD)"
    MIXAR_VERSION="$(tr -d '[:space:]' < "$MIXAR_DIR/VERSION")"
    echo "Mixar $MIXAR_VERSION at ${MIXAR_COMMIT:0:12}: $(git -C "$MIXAR_DIR" log -1 --format='%s (%cs)')"
}

apply_patches() {
    APPLIED_PATCHES=()
    [[ -f "$PATCH_SERIES" ]] || { echo "No patch series at $PATCH_SERIES"; return; }
    step "Applying patches from $PATCH_SERIES"
    local name patch optional
    while IFS= read -r name || [[ -n "$name" ]]; do
        name="${name%%#*}"
        name="$(echo "$name" | xargs)"
        [[ -n "$name" ]] || continue
        # "?name" marks a patch that may stop applying once upstream catches
        # up (e.g. ships what it adds): skipped with a warning, not fatal.
        optional=0
        if [[ "$name" == \?* ]]; then optional=1; name="${name#\?}"; fi
        patch="$(dirname "$PATCH_SERIES")/$name"
        [[ -f "$patch" ]] || die "patch listed in the series is missing: $patch"
        if ! git -C "$MIXAR_DIR" apply --check "$patch" 2>"$LOG_DIR/patch-$name.err"; then
            cat "$LOG_DIR/patch-$name.err" >&2
            if (( optional )); then
                echo "  WARNING: skipped optional $name: it no longer applies to Mixar ${MIXAR_COMMIT:0:12}"
                continue
            fi
            die "$name no longer applies to Mixar ${MIXAR_COMMIT:0:12}; rebase it onto the new version"
        fi
        git -C "$MIXAR_DIR" apply "$patch"
        rm -f "$LOG_DIR/patch-$name.err"
        APPLIED_PATCHES+=("$name")
        echo "  applied $name"
    done < "$PATCH_SERIES"
}

# Shallow-fetch one pinned commit into $1 (a repo dir), checkout without LFS.
fetch_pinned() {
    local dir="$1" url="$2" sha="$3"
    if [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" == "$sha" ]]; then
        echo "  $dir already at ${sha:0:12}"
        return
    fi
    mkdir -p "$dir"
    [[ -d "$dir/.git" ]] || git init -q "$dir"
    git -C "$dir" remote remove origin 2>/dev/null || true
    git -C "$dir" remote add origin "$url"
    git -C "$dir" fetch -q --depth 1 origin "$sha"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$dir" checkout -q -f --detach FETCH_HEAD
}

fetch_blender() {
    UPSTREAM_SHA="$(git -C "$MIXAR_DIR" ls-tree HEAD upstream | awk '{print $3}')"
    [[ -n "$UPSTREAM_SHA" ]] || die "Mixar does not pin an upstream Blender commit"
    step "Fetching Blender ${UPSTREAM_SHA:0:12} (pinned by Mixar)"
    git lfs install --skip-repo >/dev/null
    local up="$MIXAR_DIR/upstream"
    fetch_pinned "$up" "$BLENDER_GIT" "$UPSTREAM_SHA"
    # Assets, fonts and startup files live in LFS; the 800 MB of test data
    # does not, for a build, matter.
    git -C "$up" lfs pull --exclude="tests/**"
    echo "Blender: $(git -C "$up" log -1 --format='%s (%cs)')"

    LIBS_SHA="$(git -C "$up" ls-tree HEAD lib/linux_x64 | awk '{print $3}')"
    [[ -n "$LIBS_SHA" ]] || die "Blender does not pin lib/linux_x64"
    step "Fetching precompiled libraries ${LIBS_SHA:0:12}"
    # Outside upstream/ and symlinked in: the overlay rsync copies the link,
    # not 3 GB of libraries, into source/.
    local libs="$WORK_DIR/lib-linux_x64"
    if [[ -d "$libs" && ! -L "$libs" ]] && [[ "$(git -C "$libs" rev-parse HEAD 2>/dev/null || true)" != "$LIBS_SHA" ]]; then
        rm -rf "$libs"
    fi
    fetch_pinned "$libs" "$LIBS_GIT" "$LIBS_SHA"
    if [[ "$(cat "$libs/.mixar-lfs-done" 2>/dev/null || true)" != "$LIBS_SHA" ]]; then
        # dpcpp is the 800 MB oneAPI compiler, only used by the Intel GPU
        # kernels this build leaves off. Its pointer files must go too: CMake
        # globs dpcpp/lib/libsycl*.so into the bundle whether oneAPI is on or not.
        git -C "$libs" lfs pull --exclude="dpcpp/**"
        rm -rf "$libs/dpcpp" "$libs/.git/lfs/objects"
        echo "$LIBS_SHA" > "$libs/.mixar-lfs-done"
    fi
    rm -rf "$up/lib/linux_x64"
    ln -s "$libs" "$up/lib/linux_x64"
}

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
write_cmake_overrides() {
    # build.sh preloads exactly one cache file, so append to Mixar's own.
    cat >> "$MIXAR_DIR/cmake/mixar_overrides.cmake" <<EOF

# --- appended by kaggle/build_mixar_linux.sh -------------------------------
# Intel oneAPI kernels need the dpcpp compiler the build script skips.
set(WITH_CYCLES_DEVICE_ONEAPI OFF CACHE BOOL "" FORCE)
set(WITH_CYCLES_ONEAPI_BINARIES OFF CACHE BOOL "" FORCE)
set(WITH_COMPILER_CCACHE $( (( USE_CCACHE )) && echo ON || echo OFF ) CACHE BOOL "" FORCE)
EOF
}

# Keep the notebook readable: one line per percent of ninja progress, plus
# anything that looks like a failure. The full output goes to build.log.
progress_filter() {
    awk '
        /^\[[0-9]+\/[0-9]+\]/ {
            split(substr($1, 2, length($1) - 2), n, "/")
            pct = int(100 * n[1] / n[2])
            if (pct != last) { last = pct; printf("  compile %3d%%  (%s/%s)\n", pct, n[1], n[2]); fflush() }
            next
        }
        /error:|FAILED:|CMake Error|ninja: build stopped|fatal error|Traceback|ERROR:/ { print; fflush(); next }
        /^-- Configuring done|^-- Build files have been written|Overlay complete|=== Build Complete|Installing Python packages|Successfully installed|Bundled trust and MCP/ { print; fflush() }
    '
}

build_mixar() {
    step "Building Mixar $MIXAR_VERSION ($MIXAR_ENV, CUDA=$MIXAR_CUDA, $BUILD_CORES jobs) - this takes a few hours on 4 cores"
    write_cmake_overrides
    # A plain foreground pipeline: nothing is left running when it ends, so a
    # notebook cell reading this script's output always sees EOF.
    local rc=0
    (cd "$MIXAR_DIR" && CMAKE_GENERATOR=Ninja ./scripts/unix/build.sh) < /dev/null 2>&1 \
        | tee "$BUILD_LOG" | progress_filter || rc=$?
    if (( rc != 0 )); then
        echo
        echo "---- first errors in build.log ----"
        grep -n -m 40 -E 'error:|FAILED:|CMake Error|fatal error' "$BUILD_LOG" || true
        echo "---- last 80 lines of build.log ----"
        tail -n 80 "$BUILD_LOG"
        die "build.sh failed (exit $rc); full log: $BUILD_LOG"
    fi
    BIN_DIR="$MIXAR_DIR/build/$MIXAR_ENV/bin"
    [[ -x "$BIN_DIR/mixar" ]] || die "build finished but $BIN_DIR/mixar is missing"
}

smoke_test() {
    (( SMOKE_TEST )) || return 0
    step "Smoke test (headless start-up + Mixar module imports)"
    local rc=0
    timeout 900 "$BIN_DIR/mixar" -b --factory-startup --python-exit-code 3 \
        --python "$SCRIPT_DIR/smoke_test.py" > "$LOG_DIR/smoke_test.log" 2>&1 || rc=$?
    grep -E '^(SMOKE|  FAIL)' "$LOG_DIR/smoke_test.log" || true
    if (( rc != 0 )); then
        SMOKE_RESULT="failed (exit $rc, see logs/smoke_test.log)"
        echo "Smoke test $SMOKE_RESULT; packaging anyway"
    else
        SMOKE_RESULT="passed"
    fi
}

package() {
    local name="mixar-$MIXAR_VERSION-linux-x64"
    local stage_root="$WORK_DIR/pkg"
    local archive="$OUT_DIR/$name.tar.xz"
    step "Packaging $archive"
    rm -rf "$stage_root"
    mkdir -p "$stage_root"
    cp -al "$BIN_DIR" "$stage_root/$name"   # hard links: no second 2 GB copy
    {
        echo "Mixar $MIXAR_VERSION for Linux x86_64 (built from public source)"
        echo "Built:     $(date -u +%Y-%m-%dT%H:%M:%SZ)$( (( ON_KAGGLE )) && echo ' on Kaggle')"
        echo "Mixar:     $MIXAR_REPO @ $MIXAR_REF ($MIXAR_COMMIT)"
        echo "Blender:   $UPSTREAM_SHA"
        echo "Libraries: $LIBS_SHA"
        echo "Patches:   ${APPLIED_PATCHES[*]:-none}"
        echo "Compiler:  $($CXX --version | head -n1)"
        echo "Options:   MIXAR_ENV=$MIXAR_ENV MIXAR_CUDA=$MIXAR_CUDA oneAPI=OFF"
        echo "Smoke:     ${SMOKE_RESULT:-skipped}"
        echo
        echo "Run: ./mixar   (needs glibc 2.39+ / libstdc++ from GCC 14, e.g. Ubuntu 24.04+, Debian 13+, Fedora 40+)"
    } > "$stage_root/$name/BUILD_INFO.txt"
    cp "$stage_root/$name/BUILD_INFO.txt" "$OUT_DIR/BUILD_INFO.txt"
    tar -C "$stage_root" -cf - "$name" | xz -T"$BUILD_CORES" -6 > "$archive"
    (cd "$OUT_DIR" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
    rm -rf "$stage_root"
    zstd -q -f -19 "$BUILD_LOG" -o "$BUILD_LOG.zst" && rm -f "$BUILD_LOG"
    ls -lh "$archive"
}

main() {
    print_system
    (( SKIP_DEPS )) || install_deps
    select_compiler
    resolve_mixar_ref
    fetch_mixar
    apply_patches
    fetch_blender
    df -h "$WORK_DIR" | tail -n 1
    build_mixar
    smoke_test
    package
    step "Done in $(( ($(date +%s) - START_TS) / 60 )) min"
    cat "$OUT_DIR/BUILD_INFO.txt"
}

main "$@"
