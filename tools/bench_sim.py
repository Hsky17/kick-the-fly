"""Sim-time vs real-time for N independent fly brains, run the way the game runs them (one Brain thread per fly).

    python tools/bench_sim.py --flies 1 8 --seconds 10

Reports, per fly count: steps/s per brain when paced at real time (200 steps/s = 1.00x keeps up), and the uncapped
throughput (how fast each brain could run if it didn't wait for the clock).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import brainpack  # noqa: E402
import kick_the_fly as k  # noqa: E402
from connectome.sim import LIFParams, LIFSim  # noqa: E402


def make(g, W, seed):
    sim = LIFSim(None, LIFParams(), W_in=W, seed=seed)
    br = k.Brain(g, sim, seed=seed)
    if getattr(g, "dan_mbon", None) is not None:
        import memory
        os.environ.setdefault("KICK_THE_FLY_MEMORY", str(Path(os.environ.get("TMPDIR", "/tmp")) / "ktf-bench-mem"))
        br.memory = memory.Memory(g, sim)
        br.memory.save = lambda: None
    br.warmup(200)
    return br


def measure(brains, seconds, speed):
    for b in brains:
        b.speed = speed
    s0 = [b.steps for b in brains]
    t0 = time.perf_counter()
    for b in brains:
        b.start()
    time.sleep(seconds)
    dt = time.perf_counter() - t0
    for b in brains:
        b.stop()
    time.sleep(0.2)
    return [(b.steps - s) / dt for b, s in zip(brains, s0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flies", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    g, W, _ = brainpack.load(brainpack.find())
    for n in args.flies:
        brains = [make(g, W, seed) for seed in range(n)]
        paced = measure(brains, args.seconds, 1.0)
        for b in brains:
            b._stop = False
        uncapped = measure(brains, args.seconds, 1000.0)
        rt = 1000.0 / brains[0].sim.p.dt_ms
        print(f"flies={n:2d}  paced: {np.mean(paced):6.1f} steps/s per brain = {np.mean(paced) / rt:4.2f}x real time "
              f"(min {np.min(paced) / rt:4.2f}x)   uncapped: {np.mean(uncapped):6.1f} steps/s = {np.mean(uncapped) / rt:4.2f}x",
              flush=True)
        del brains
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
