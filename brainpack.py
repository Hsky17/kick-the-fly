"""Compact brain pack for Kick the Fly: everything the game needs from the connectome in one ~40 MB file.

Holds the signed synapse matrix [post, pre] as int16 counts plus each neuron's 1 / total input synapses, the
neuron type, superclass and instance labels, and each neuron's cell-body position. The game rebuilds the
simulator's rate-normalized matrix from it in about a second, so a packaged build skips the 1.1 GB download.

Derived from Janelia FlyEM MaleCNS v1.0 (CC BY 4.0).

    python brainpack.py build      # data/graph.pkl + body annotations -> data/kick_brain.npz
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

PACK_NAME = "kick_brain.npz"
DATA_DIR = Path(__file__).resolve().parent / "data"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"


def soma_positions(g) -> np.ndarray:
    """(n, 3) float32 cell-body position per graph row, NaN where the annotation has none (e.g. sensory neurons)."""
    import pyarrow.feather as feather

    t = feather.read_table(DATA_DIR / ANNOTATIONS, columns=["bodyId", "somaLocation", "tosomaLocation"])
    rows = np.array([g.index.get(int(b), -1) for b in t["bodyId"].to_numpy()])
    pos = np.full((g.n, 3), np.nan, np.float32)
    for col in ("tosomaLocation", "somaLocation"):          # a real soma wins over the soma-tract estimate
        for r, v in zip(rows, t[col].to_pylist()):
            if r >= 0 and v is not None:
                pos[r] = v
    return pos


def build(out: Path = DATA_DIR / PACK_NAME) -> Path:
    from connectome.loader import load_graph

    g = load_graph()
    signed = g.adjacency.T.tocsr()                          # [post, pre]
    counts_in = np.asarray(g.weights.sum(axis=0)).ravel()
    inv = np.where(counts_in > 0, 1.0 / np.maximum(counts_in, 1), 0).astype(np.float32)
    signed.eliminate_zeros()
    if np.abs(signed.data).max() > np.iinfo(np.int16).max:
        raise ValueError("synapse count does not fit int16")

    def labels(a):
        return np.array(["" if x is None else str(x) for x in a])

    soma = soma_positions(g)
    np.savez_compressed(
        out, indptr=signed.indptr.astype(np.int32), indices=signed.indices.astype(np.int32),
        data=signed.data.astype(np.int16), inv=inv, type=labels(g.type), superclass=labels(g.superclass),
        instance=labels(g.instance), soma=soma,
    )
    print(f"[brainpack] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {g.n:,} neurons, {signed.nnz:,} synapse pairs, "
          f"{int((~np.isnan(soma[:, 0])).sum()):,} cell bodies)")
    return out


def find() -> Path | None:
    """The pack next to a PyInstaller bundle, next to the exe, or in data/."""
    roots = [Path(getattr(sys, "_MEIPASS", "")), Path(sys.executable).resolve().parent, DATA_DIR]
    for root in roots:
        if str(root) and (root / PACK_NAME).exists():
            return root / PACK_NAME
    return None


def load(path: Path):
    """Returns (graph-like namespace with n/type/superclass/instance, W_in [post, pre] float32, soma (n, 3))."""
    z = np.load(path)
    inv = z["inv"]
    n = len(inv)
    data = z["data"].astype(np.float32)
    indptr = z["indptr"]
    data *= np.repeat(inv, np.diff(indptr))                 # row-normalize by the post neuron's input count
    W = sp.csr_array((data, z["indices"], indptr), shape=(n, n))
    inst = z["instance"]
    g = SimpleNamespace(n=n, type=z["type"], superclass=z["superclass"], instance=np.where(inst == "", None, inst))
    return g, W, z["soma"]


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        print(__doc__)
