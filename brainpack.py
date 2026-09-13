"""Compact brain pack for Kick the Fly: everything the game needs from the connectome in one ~50 MB file.

Holds the signed synapse matrix [post, pre] as int16 counts plus each neuron's 1 / total input synapses, the
neuron type, superclass and instance labels, and the 2D layout. The game rebuilds the simulator's rate-normalized
matrix from it in about a second, so a packaged build skips the 1.1 GB download and the graph and layout builds.

Derived from Janelia FlyEM MaleCNS v1.0 (CC BY 4.0).

    python brainpack.py build      # data/graph.pkl + data/layout.npy -> data/kick_brain.npz
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

PACK_NAME = "kick_brain.npz"
DATA_DIR = Path(__file__).resolve().parent / "data"


def build(out: Path = DATA_DIR / PACK_NAME) -> Path:
    from connectome.layout import load_layout
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

    np.savez_compressed(
        out, indptr=signed.indptr.astype(np.int32), indices=signed.indices.astype(np.int32),
        data=signed.data.astype(np.int16), inv=inv, type=labels(g.type), superclass=labels(g.superclass),
        instance=labels(g.instance), layout=load_layout(g).astype(np.float32),
    )
    print(f"[brainpack] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {g.n:,} neurons, {signed.nnz:,} synapse pairs)")
    return out


def find() -> Path | None:
    """The pack next to a PyInstaller bundle, next to the exe, or in data/."""
    roots = [Path(getattr(sys, "_MEIPASS", "")), Path(sys.executable).resolve().parent, DATA_DIR]
    for root in roots:
        if str(root) and (root / PACK_NAME).exists():
            return root / PACK_NAME
    return None


def load(path: Path):
    """Returns (graph-like namespace with n/type/superclass/instance, W_in [post, pre] float32, layout)."""
    z = np.load(path)
    inv = z["inv"]
    n = len(inv)
    data = z["data"].astype(np.float32)
    indptr = z["indptr"]
    data *= np.repeat(inv, np.diff(indptr))                 # row-normalize by the post neuron's input count
    W = sp.csr_array((data, z["indices"], indptr), shape=(n, n))
    inst = z["instance"]
    g = SimpleNamespace(n=n, type=z["type"], superclass=z["superclass"], instance=np.where(inst == "", None, inst))
    return g, W, z["layout"]


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        print(__doc__)
