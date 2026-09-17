#!/bin/sh
# Runs the release AppImage. It mounts itself with FUSE when it can; otherwise it extracts itself to a temporary
# folder and runs from there, so the game works without FUSE too.
APPIMAGE_FILE=/opt/kickthefly/KickTheFly-x86_64.AppImage
if [ -e /dev/fuse ] && { command -v fusermount >/dev/null 2>&1 || command -v fusermount3 >/dev/null 2>&1; }; then
    exec "$APPIMAGE_FILE" "$@"
fi
exec "$APPIMAGE_FILE" --appimage-extract-and-run "$@"
