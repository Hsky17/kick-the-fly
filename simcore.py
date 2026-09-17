"""Brains without a window: build, drive and step LIF brains synchronously ("lockstep").

The game runs each brain on its own real-time thread. Assays, validation, protocols, save-state tests and repeated
trials instead step brains directly from the calling thread, so a run is exactly reproducible: the same seed and the
same stimulus schedule give the same spikes on the same machine.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@lru_cache(maxsize=1)
def pack():
    """(graph-like namespace, W_in, soma) from the bundled brain pack, loaded once per process."""
    import brainpack

    path = brainpack.find()
    if path is None:
        raise FileNotFoundError("brain pack kick_brain.npz not found (run 'python brainpack.py build')")
    return brainpack.load(path)


def new_brain(seed: int = 0, memory: bool = True, warmup: int = 600, params: dict | None = None,
              isolated_memory: bool = True):
    """A warmed-up Brain that is not running on a thread. isolated_memory: start from the untrained connectome and never
    read or write the player's saved training memory."""
    import kick_the_fly as k
    import lab
    from connectome.sim import LIFParams, LIFSim

    g, W, _ = pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=seed)
    if params:
        lab.apply_to_sim(sim, params)
    br = k.Brain(g, sim, seed=seed)
    if memory and getattr(g, "dan_mbon", None) is not None:
        import memory as mem_mod

        br.memory = mem_mod.Memory(g, sim, load=not isolated_memory)
        br.memory.save = lambda: None
    if warmup:
        br.warmup(warmup)
    return br


def step(br, n: int, record: np.ndarray | None = None) -> np.ndarray | None:
    """Advance n steps. record: neuron rows whose spikes to return as an (n, len(rows)) bool array."""
    out = None if record is None else np.zeros((n, len(record)), bool)
    for i in range(n):
        br._step()
        if out is not None:
            out[i] = br.sim.spikes[record]
    return out


def rows_of(br, spec) -> np.ndarray:
    """Neuron rows from a spec: a group name (e.g. 'loom', 'escape'), 'type:DNp01,MDN', 'prefix:KC', 'superclass:x'
    or 'rows:1,2,3'."""
    if isinstance(spec, (list, tuple, np.ndarray)):
        return np.asarray(spec, np.int64)
    spec = str(spec)
    if spec.startswith("type:"):
        return np.flatnonzero(np.isin(br.types, spec[5:].split(",")))
    if spec.startswith("prefix:"):
        m = np.zeros(br.n, bool)
        for p in spec[7:].split(","):
            m |= np.char.startswith(br.types, p)
        return np.flatnonzero(m)
    if spec.startswith("superclass:"):
        return np.flatnonzero(np.isin(br.superclass, spec[11:].split(",")))
    if spec.startswith("rows:"):
        return np.array([int(x) for x in spec[5:].split(",") if x], np.int64)
    if spec in br.col:
        i = br.col[spec]
        if i < br.n_det:
            return np.flatnonzero(br.det_id == i)
        if spec == "whole brain":
            return np.arange(br.n)
        return np.flatnonzero(br.pop_id == i - br.n_det)
    raise ValueError(f"unknown neuron spec {spec!r}")


def drive(br, rows: np.ndarray, amp: float = 0.5) -> None:
    """Hold these neurons driven (like brain surgery's ON, with a chosen current) until undrive()."""
    br.override[rows] = amp
    br.surgery = bool(np.any(br.override))


def undrive(br, rows: np.ndarray) -> None:
    br.override[rows] = 0
    br.surgery = bool(np.any(br.override))
