"""Headless runs: no window, no sound, no display needed (SSH, CI, a Windows console).

    KickTheFly --headless --validate [--out results.json] [--workers N] [--seeds 1000-1009]
    KickTheFly --headless --protocol experiment.yaml [--out folder]

The exe and the AppImage accept the same flags. On Windows the exe borrows the console it was started from for its
output; from cmd use `start /wait KickTheFly.exe --headless ...` (or check the files written to --out).
Exit codes: 0 success; 1 a validation result differs from the expected pass/fail (only with --strict); 2 bad input.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from crash import log


def prepare() -> None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"          # nothing in a headless run may open a window or an audio device
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import platform_env

    platform_env.attach_console()


def parse_seeds(text: str | None, default):
    if not text:
        return tuple(default)
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return tuple(out)


def run_validate(args) -> int:
    import validation

    seeds = parse_seeds(args.seeds, validation.SEEDS)
    t0 = time.time()

    def progress(done, total, label):
        print(f"  {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    res = validation.run(seeds=seeds, workers=args.workers, progress=progress)
    out = Path(args.out) if args.out else validation.local_path()
    if out.suffix.lower() != ".json":
        out = out / validation.RESULTS_NAME
    validation.save_results(res, out)
    print(validation.summary(res))
    print(f"results written to {out}")
    if args.strict:
        wrong = [t["id"] for t in res["tests"] if t["passed"] != validation.EXPECTED.get(t["id"])]
        if wrong:
            print(f"differs from the expected results: {', '.join(wrong)}")
            return 1
    return 0


def main(args) -> int:
    prepare()
    log.info("headless run")
    try:
        if args.validate:
            return run_validate(args)
        if args.protocol:
            import protocol

            return protocol.run_file(Path(args.protocol), Path(args.out) if args.out else None, workers=args.workers)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("nothing to do: use --validate or --protocol FILE", file=sys.stderr)
    return 2
