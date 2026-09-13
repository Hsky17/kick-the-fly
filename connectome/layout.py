"""2D neuron layout for the activity panel, computed once and cached.

Symmetrized log1p(|synapse count|) adjacency -> TruncatedSVD (32 dims) ->
UMAP (cosine) to 2D, or PCA of the SVD embedding with --method pca.
Cached to data/layout.npy as float32 (n, 2) in [0, 1].

    python -m connectome.layout build [--method umap|pca]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from connectome.loader import DATA_DIR, Graph, load_graph

LAYOUT_PATH = DATA_DIR / "layout.npy"


def build_layout(g: Graph, method: str = "umap", seed: int = 0) -> np.ndarray:
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    t0 = time.perf_counter()
    A = g.weights.astype(np.float32)
    A = (A + A.T).tocsr()
    A.data = np.log1p(A.data)
    emb = TruncatedSVD(n_components=32, random_state=seed).fit_transform(A)
    emb = normalize(emb)
    print(f"[layout] TruncatedSVD(32) on {A.nnz:,} nnz in {time.perf_counter() - t0:.1f}s", flush=True)

    if method == "umap":
        import umap

        t = time.perf_counter()
        xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05, metric="cosine",
                       low_memory=True, random_state=seed, verbose=False).fit_transform(emb)
        print(f"[layout] UMAP in {time.perf_counter() - t:.1f}s", flush=True)
    else:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=seed).fit_transform(emb)

    xy = xy.astype(np.float32)
    lo, hi = np.percentile(xy, [0.5, 99.5], axis=0)
    xy = np.clip((xy - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    np.save(LAYOUT_PATH, xy)
    print(f"[layout] wrote {LAYOUT_PATH} ({method}, {len(xy):,} points) total {time.perf_counter() - t0:.1f}s")
    return xy


def load_layout(g: Graph) -> np.ndarray:
    if not LAYOUT_PATH.exists():
        raise FileNotFoundError(f"{LAYOUT_PATH} not found. Build it once with: python -m connectome.layout build")
    xy = np.load(LAYOUT_PATH)
    if len(xy) != g.n:
        raise RuntimeError(f"{LAYOUT_PATH} has {len(xy):,} points but graph has {g.n:,}; rebuild the layout")
    return xy


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m connectome.layout")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--method", choices=["umap", "pca"], default="umap")
    args = parser.parse_args(argv)
    build_layout(load_graph(), args.method)
    return 0


if __name__ == "__main__":
    from connectome.layout import main as _main
    raise SystemExit(_main())
