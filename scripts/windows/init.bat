REM SPDX-FileCopyrightText: 2026 Adeveda Enterprises Private Limited
REM
REM SPDX-License-Identifier: GPL-2.0-or-later

@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Initialize Blender to the exact commit specified by the parent repository.
echo Initializing Blender submodule...
set "GIT_LFS_SKIP_SMUDGE=1"
git submodule update --init --recursive --force --progress
if errorlevel 1 (
    echo Error: Failed to initialize Blender submodule.
    exit /b 1
)

if not exist "upstream\CMakeLists.txt" (
    echo Error: Blender upstream checkout is missing or incomplete.
    exit /b 1
)

REM Blender's Windows bootstrap asks interactively whether lib/windows_x64
REM should be downloaded when it is missing. GitHub Actions has no interactive
REM stdin, so the prompt is interpreted as "no" and make update exits 1.
REM Pre-initialize exactly the same dependency submodule as Blender's
REM build_files/windows/check_libraries.cmd, then make update is non-interactive.
echo Preparing Blender Windows x64 precompiled libraries...
git -C upstream config --local "submodule.lib/windows_x64.update" "checkout"
if errorlevel 1 (
    echo Error: Failed to configure lib/windows_x64 submodule checkout.
    exit /b 1
)

git -C upstream submodule update --progress --init "lib/windows_x64"
if errorlevel 1 (
    echo Error: Failed to initialize Blender lib/windows_x64.
    exit /b 1
)

REM The dependency repository is LFS-backed. Skip smudge during checkout for
REM reliability, then explicitly download all LFS objects once the checkout is ready.
set "GIT_LFS_SKIP_SMUDGE="
git -C "upstream\lib\windows_x64" lfs pull
if errorlevel 1 (
    echo Error: Failed to download Blender lib/windows_x64 LFS objects.
    exit /b 1
)

if not exist "upstream\lib\windows_x64\.git" (
    echo Error: Blender Windows x64 libraries were not initialized correctly.
    exit /b 1
)

pushd upstream
if errorlevel 1 (
    echo Error: Failed to enter Blender upstream directory.
    exit /b 1
)

REM Update Blender source/submodules now that the required Windows library
REM checkout already exists, preventing check_libraries.cmd from prompting.
echo Running make update...
make update
if errorlevel 1 (
    echo Error: Blender make update failed.
    popd
    exit /b 1
)

echo Pulling Blender source LFS files...
git lfs pull
if errorlevel 1 (
    echo Error: Blender source git lfs pull failed.
    popd
    exit /b 1
)

popd
echo Initialization complete!
exit /b 0
