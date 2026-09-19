"""Arbitrary-duration MP4/WebM video recording with ffmpeg detection and GIF fallback."""
from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pygame

log = logging.getLogger(__name__)


def ffmpeg_available() -> bool:
    """Return True if ffmpeg executable is discovered on system PATH."""
    return bool(shutil.which("ffmpeg"))


class VideoRecorder:
    """Streams frames to ffmpeg via stdin pipe for arbitrary duration recording.

    Falls back cleanly to animated GIF export via Pillow if ffmpeg is absent.
    """

    def __init__(self) -> None:
        self.is_recording: bool = False
        self.output_path: Path | None = None
        self.start_time: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.fps: int = 30
        self.use_ffmpeg: bool = False
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[bytes | None] | None = None
        self._worker: threading.Thread | None = None
        self._gif_frames: list[bytes] = []
        self._frames_written: int = 0
        self._error: str | None = None

    def start(
        self,
        output_path: str | Path | None = None,
        width: int = 1280,
        height: int = 760,
        fps: int = 30,
    ) -> Path:
        """Start recording frames to video file or GIF fallback."""
        if self.is_recording:
            assert self.output_path is not None
            return self.output_path

        has_ffmpeg = ffmpeg_available()

        # Resolve destination path
        if output_path is None or output_path == "default":
            vid_dir = Path.cwd() / "videos"
            vid_dir.mkdir(parents=True, exist_ok=True)
            stem = f"kick-the-fly-{time.strftime('%Y%m%d-%H%M%S')}"
            ext = "mp4" if has_ffmpeg else "gif"
            target = vid_dir / f"{stem}.{ext}"
            n = 2
            while target.exists():
                target = vid_dir / f"{stem}-{n}.{ext}"
                n += 1
        else:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.suffix:
                target = target.with_suffix(".mp4" if has_ffmpeg else ".gif")

        # H.264 and VP9 require even dimensions
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.fps = max(1, int(fps))
        self.start_time = time.perf_counter()
        self._frames_written = 0
        self._error = None

        ext = target.suffix.lower().lstrip(".")
        self.use_ffmpeg = has_ffmpeg and ext in ("mp4", "webm", "mkv", "mov")

        if self.use_ffmpeg:
            self.output_path = target
            if ext == "webm":
                vcodec_args = ["-c:v", "libvpx-vp9", "-b:v", "2M", "-pix_fmt", "yuv420p"]
            else:
                vcodec_args = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.width}x{self.height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(self.fps),
                "-i",
                "-",
                *vcodec_args,
                str(self.output_path),
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._queue = queue.Queue(maxsize=180)
                self._worker = threading.Thread(target=self._pipe_worker, daemon=True)
                self._worker.start()
            except Exception as e:
                log.warning("Failed to start ffmpeg subprocess: %s, falling back to GIF", e)
                self.use_ffmpeg = False
                self._gif_frames = []
                self.output_path = target.with_suffix(".gif")
        else:
            self._gif_frames = []
            self.output_path = target.with_suffix(".gif") if ext != "gif" else target

        self.is_recording = True
        return self.output_path

    def _pipe_worker(self) -> None:
        try:
            while True:
                assert self._queue is not None
                item = self._queue.get()
                if item is None:
                    break
                if self._proc and self._proc.stdin:
                    try:
                        self._proc.stdin.write(item)
                    except (BrokenPipeError, OSError) as e:
                        self._error = str(e)
                        break
                self._queue.task_done()
        except Exception as e:
            self._error = str(e)

    def write_frame_bytes(self, raw_rgb: bytes) -> None:
        """Write raw RGB byte array of size (width * height * 3)."""
        if not self.is_recording:
            return
        if self.use_ffmpeg and self._queue is not None:
            try:
                self._queue.put_nowait(raw_rgb)
                self._frames_written += 1
            except queue.Full:
                # Drop frame under extreme load rather than blocking game loop
                pass
        else:
            # GIF mode: sample at ~15 fps to keep file size reasonable
            stride = max(1, self.fps // 15)
            if self._frames_written % stride == 0:
                self._gif_frames.append(raw_rgb)
            self._frames_written += 1

    def write_frame_surface(self, surf: pygame.Surface) -> None:
        """Write pygame.Surface frame, rescaling to target resolution if needed."""
        if not self.is_recording:
            return
        w, h = surf.get_size()
        if w != self.width or h != self.height:
            surf = pygame.transform.smoothscale(surf, (self.width, self.height))
        raw = pygame.image.tobytes(surf, "RGB")
        self.write_frame_bytes(raw)

    def elapsed(self, now: float | None = None) -> float:
        """Elapsed recording duration in seconds."""
        if not self.is_recording:
            return 0.0
        t = time.perf_counter() if now is None else now
        return max(0.0, t - self.start_time)

    def badge_text(self, now: float | None = None) -> str:
        """Formatted HUD badge, e.g. '[REC 01:23]'."""
        el = int(self.elapsed(now))
        m, s = divmod(el, 60)
        return f"[REC {m:02d}:{s:02d}]"

    def stop(self) -> Path | None:
        """Stop recording, flush stream or finalize GIF, and return saved path."""
        if not self.is_recording:
            return None
        self.is_recording = False
        out = self.output_path

        if self.use_ffmpeg and self._proc:
            if self._queue:
                self._queue.put(None)
            if self._worker:
                self._worker.join(timeout=10.0)
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except OSError:
                    pass
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            self._queue = None
            self._worker = None
        elif self._gif_frames:
            try:
                from PIL import Image

                size = (self.width, self.height)
                sub = self._gif_frames
                if sub:
                    imgs = [
                        Image.frombytes("RGB", size, f).convert("P", palette=Image.ADAPTIVE, colors=160)
                        for f in sub
                    ]
                    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=66, loop=0)
            except Exception as e:
                log.error("Failed to save GIF fallback: %s", e)
            self._gif_frames.clear()

        return out
