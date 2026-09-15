#!/usr/bin/env bash
# Builds dist/KickTheFly-x86_64.AppImage: one file with the game, Python, numpy/scipy/pygame and the brain pack inside.
# Needs data/kick_brain.npz first:  python brainpack.py build   (after the connectome/layout builds in README.md)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f data/kick_brain.npz ]; then
    echo "data/kick_brain.npz missing: run 'python brainpack.py build' first" >&2
    exit 1
fi

PY=$(command -v python3.11 || command -v python3)

if [ ! -d .venv-build ]; then
    "$PY" -m venv .venv-build
    .venv-build/bin/pip install --upgrade pip
    .venv-build/bin/pip install numpy scipy pygame pyinstaller pillow moderngl
fi

.venv-build/bin/python tools/make_icon.py build/icon.png

# glcontext.x11/egl are moderngl's Linux GL backends (glcontext.wgl is the Windows-only one build_exe.ps1 uses).
.venv-build/bin/pyinstaller --noconfirm --clean --onefile --name KickTheFly \
    --add-data "data/kick_brain.npz:." \
    --exclude-module connectome.loader --exclude-module connectome.layout --exclude-module pyarrow \
    --exclude-module tkinter --exclude-module matplotlib \
    --hidden-import kick3d --hidden-import render3d --hidden-import memory \
    --hidden-import glcontext.x11 --hidden-import glcontext.egl --hidden-import glcontext.empty \
    kick_the_fly.py

dist/KickTheFly --smoke 3 >/dev/null

# --- assemble the AppDir ---
APPDIR=build/KickTheFly.AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp dist/KickTheFly "$APPDIR/usr/bin/KickTheFly"
cp build/icon.png "$APPDIR/kickthefly.png"

cat > "$APPDIR/kickthefly.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Kick the Fly
Comment=Kick a real fruit fly connectome
Exec=KickTheFly
Icon=kickthefly
Categories=Game;
Terminal=false
EOF

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/KickTheFly" "$@"
EOF
chmod +x "$APPDIR/AppRun"

APPIMAGETOOL=build/appimagetool-x86_64.AppImage
if [ ! -f "$APPIMAGETOOL" ]; then
    curl -L -o "$APPIMAGETOOL" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$APPIMAGETOOL"
fi

mkdir -p dist
rm -f dist/KickTheFly-x86_64.AppImage
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" dist/KickTheFly-x86_64.AppImage
ls -lh dist/KickTheFly-x86_64.AppImage
