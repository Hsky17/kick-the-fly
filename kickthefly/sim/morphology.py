"""Real neuron morphology loader with neuPrint skeleton fetching and local caching.

Fetches real EM reconstruction skeletons (SWC format) from Janelia neuPrint (MaleCNS v1.0)
for key neuron classes:
  - Giant Fiber escape command neurons (DNp01)
  - Descending steering neurons (DNa02)
  - Mushroom body output neurons (MBONs)
  - Kenyon cells (KCs)

Features:
  - Local caching in data/skeletons/ so skeletons are only downloaded once.
  - Graceful fallback to synthetic fiber approximations if neuPrint is offline or unreachable.
  - Strict rate-limiting to respect server quotas and prevent 429/bans.
  - SWC skeleton parsing and resampling into 3D point arrays matching BrainView render buffers.
"""
from __future__ import annotations

import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np

log = logging.getLogger("kickthefly")

NEUPRINT_BASE_URL = "https://neuprint.janelia.org/api/skeletons/skeleton"
DATASET = "male-cns:v1.0"
KEY_TYPES = ("DNp01", "DNa02", "MBON01", "MBON14", "KCg")
RATE_LIMIT_S = 0.25
_last_request_time = 0.0


def default_cache_dir() -> Path:
    """Path to local skeleton cache in data/skeletons or user state directory."""
    repo_cache = Path(__file__).resolve().parent.parent.parent / "data" / "skeletons"
    try:
        repo_cache.mkdir(parents=True, exist_ok=True)
        return repo_cache
    except OSError:
        from kickthefly.core import paths
        p = paths.get().cache_dir / "skeletons"
        p.mkdir(parents=True, exist_ok=True)
        return p


def parse_swc(text: str, n_samples: int = 21) -> np.ndarray | None:
    """Parses SWC format text into an (n_samples, 3) float32 array of coordinates."""
    nodes = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                nodes.append([float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue

    if len(nodes) < 2:
        return None

    coords = np.array(nodes, dtype=np.float32)
    if len(coords) == n_samples:
        return coords

    # Evenly sample indices along the reconstruction graph / trajectory
    indices = np.linspace(0, len(coords) - 1, n_samples).astype(np.int64)
    return coords[indices]


def fetch_or_load_skeleton(body_id: int, cache_dir: Path | None = None,
                           allow_network: bool = True) -> np.ndarray | None:
    """Returns (n_samples, 3) coordinates for a body_id, checking cache first then neuPrint."""
    global _last_request_time
    if cache_dir is None:
        cache_dir = default_cache_dir()

    cache_file = cache_dir / f"{body_id}.swc"
    if cache_file.exists():
        try:
            return parse_swc(cache_file.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            log.warning("Failed to read cached skeleton %s: %s", cache_file, e)

    if not allow_network:
        return None

    # Rate limiting
    now = time.perf_counter()
    elapsed = now - _last_request_time
    if elapsed < RATE_LIMIT_S:
        time.sleep(RATE_LIMIT_S - elapsed)

    url = f"{NEUPRINT_BASE_URL}/{DATASET}/{body_id}?format=swc"
    req = urllib.request.Request(url, headers={"User-Agent": "KickTheFly/2.8 (connectome-research)"})
    try:
        _last_request_time = time.perf_counter()
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            content = resp.read()
            try:
                cache_file.write_bytes(content)
            except OSError:
                pass
            return parse_swc(content.decode("utf-8", errors="ignore"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        log.info("neuPrint skeleton fetch failed for body %d (%s); using synthetic fibers", body_id, e)
        return None


def load_key_skeletons(graph, cache_dir: Path | None = None,
                       allow_network: bool = True) -> tuple[dict[int, np.ndarray], str]:
    """Loads skeletons for key neurons (DNp01, DNa02, MBONs, KCs).

    Returns:
        (skeletons_by_idx, status_message)
    """
    if graph is None or getattr(graph, "body_id", None) is None:
        return {}, "Skeletons offline - using synthetic fibers"

    types = graph.type.astype(str)
    bodies = graph.body_id
    skeletons: dict[int, np.ndarray] = {}
    network_failed = False

    for t_name in KEY_TYPES:
        idxs = np.flatnonzero(np.char.startswith(types, t_name))
        # Pick up to 2 instances per type for crisp representation
        for idx in idxs[:2]:
            bid = int(bodies[idx])
            if bid <= 0:
                continue
            coords = fetch_or_load_skeleton(bid, cache_dir=cache_dir, allow_network=allow_network and not network_failed)
            if coords is not None:
                skeletons[int(idx)] = coords
            else:
                if allow_network:
                    network_failed = True

    if skeletons:
        status = f"Real morphology: {len(skeletons)} neuPrint skeletons active"
    else:
        status = "Skeletons offline - using synthetic fibers"

    return skeletons, status
