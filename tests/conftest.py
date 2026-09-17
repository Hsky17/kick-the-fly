import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test writes config, memory, saves and pictures under a temp folder, never the real user folders."""
    monkeypatch.setenv("KICK_THE_FLY_HOME", str(tmp_path / "ktf-home"))
    monkeypatch.delenv("KICK_THE_FLY_MEMORY", raising=False)
    import paths
    paths.reset_cache()
    yield tmp_path / "ktf-home"
    paths.reset_cache()


def brain_pack():
    import brainpack
    return brainpack.find()


needs_pack = pytest.mark.skipif(brain_pack() is None, reason="brain pack data/kick_brain.npz not built")
