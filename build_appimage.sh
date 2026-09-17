#!/usr/bin/env bash
# Builds dist/KickTheFly-x86_64.AppImage: one file with the game, Python, numpy/scipy/pygame and the brain pack inside.
# Needs data/kick_brain.npz first:  python brainpack.py build   (after the connectome build in README.md)
# Build on an old glibc (the release uses ubuntu-22.04) so the AppImage runs on most distros.
# data/validation_results.json, if present, is bundled so the real-science popups work (python kick_the_fly.py
# --headless --validate --out data/validation_results.json).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
APP_ID=io.github.legendarylolo318_cloud.KickTheFly
VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' version.py)

if [ ! -f data/kick_brain.npz ]; then
    echo "data/kick_brain.npz missing: run 'python brainpack.py build' first" >&2
    exit 1
fi

PY=${PYTHON:-$(command -v python3.11 || command -v python3)}

if [ ! -d .venv-build ]; then
    "$PY" -m venv .venv-build
    .venv-build/bin/pip install --upgrade pip
fi
.venv-build/bin/pip install numpy scipy pygame pyinstaller pillow moderngl pyyaml

.venv-build/bin/python tools/make_icon.py build/icon.png

EXTRA_DATA=(--add-data "data/kick_brain.npz:." --add-data "protocols:protocols")
if [ -f data/validation_results.json ]; then
    EXTRA_DATA+=(--add-data "data/validation_results.json:.")
else
    echo "note: data/validation_results.json not found; the build won't show real-science popups" >&2
fi

# glcontext.x11/egl are moderngl's Linux GL backends (glcontext.wgl is the Windows-only one build_exe.ps1 uses).
.venv-build/bin/pyinstaller --noconfirm --clean --onefile --name KickTheFly \
    "${EXTRA_DATA[@]}" \
    --exclude-module connectome.loader --exclude-module connectome.layout --exclude-module pyarrow \
    --exclude-module tkinter --exclude-module matplotlib \
    --hidden-import kick3d --hidden-import render3d --hidden-import memory \
    --hidden-import glcontext.x11 --hidden-import glcontext.egl --hidden-import glcontext.empty \
    --hidden-import assays --hidden-import challenges --hidden-import validation --hidden-import simcore \
    --hidden-import savestate --hidden-import recorder --hidden-import protocol --hidden-import headless \
    --hidden-import labjobs --hidden-import labstats --hidden-import lab --hidden-import yaml \
    kick_the_fly.py

# build check without a display: load the brain, run the smoke protocol, exit 0
rm -rf build/smoke
dist/KickTheFly --headless --protocol protocols/smoke.yaml --out build/smoke
ls build/smoke/*/summary.json >/dev/null

# --- assemble the AppDir ---
APPDIR=build/KickTheFly.AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/metainfo" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp dist/KickTheFly "$APPDIR/usr/bin/KickTheFly"
cp build/icon.png "$APPDIR/$APP_ID.png"
cp build/icon.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png"
cp "packaging/linux/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
cp "packaging/linux/$APP_ID.desktop" "$APPDIR/usr/share/applications/$APP_ID.desktop"
cp "packaging/linux/$APP_ID.appdata.xml" "$APPDIR/usr/share/metainfo/$APP_ID.appdata.xml"

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/KickTheFly" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

APPIMAGETOOL=build/appimagetool-x86_64.AppImage
if [ ! -f "$APPIMAGETOOL" ]; then
    curl -fL -o "$APPIMAGETOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$APPIMAGETOOL"
fi

mkdir -p dist
rm -f dist/KickTheFly-x86_64.AppImage
# APPIMAGE_EXTRACT_AND_RUN: appimagetool is itself an AppImage; this runs it without FUSE (containers, CI)
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 VERSION="$VERSION" "$APPIMAGETOOL" "$APPDIR" dist/KickTheFly-x86_64.AppImage
ls -lh dist/KickTheFly-x86_64.AppImage
