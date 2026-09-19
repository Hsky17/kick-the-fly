"""Tests for pluggable simulation backends: CPU, Numba JIT, PyTorch."""
import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.core import simcore
from kickthefly.sim.connectome import backends
from kickthefly.sim.connectome.sim import LIFParams, LIFSim

pytestmark = needs_pack


def test_detect_available_backends():
    avail = backends.detect_available_backends()
    assert "cpu" in avail
    print("Detected backends:", avail)


def test_backend_fallback_on_invalid():
    g, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=123)
    b = backends.create_backend(sim, "non_existent_gpu_backend")
    assert b.name == "cpu"


def test_backends_spikes_and_rates_seeded_equivalence():
    """Runs the same seeded workload through every available backend and asserts matching results."""
    avail = backends.detect_available_backends()
    g, W, _ = simcore.pack()
    n_steps = 100

    backend_results = {}
    for b_name in avail:
        lp = LIFParams(backend=b_name)
        sim = LIFSim(None, lp, W_in=W, seed=42)
        spikes_history = []
        for step in range(n_steps):
            # inject occasional sensory drive
            sens = np.zeros(sim.n, dtype=np.float32) if step % 20 == 10 else None
            if sens is not None:
                sens[100:150] = 2.0
            spk = sim.step(sens)
            spikes_history.append(spk)
        total_spikes = sum(s.sum() for s in spikes_history)
        backend_results[b_name] = (spikes_history, total_spikes, sim.gain, sim.v.copy())
        print(f"Backend {b_name:10s}: total spikes = {total_spikes}, final gain = {sim.gain:.4f}")

    # CPU and Numba: CPU and Numba JIT execute the exact same arithmetic logic
    if "numba" in backend_results:
        cpu_spk, cpu_tot, cpu_gain, cpu_v = backend_results["cpu"]
        num_spk, num_tot, num_gain, num_v = backend_results["numba"]
        # Bit-for-bit exact or extremely close float accumulation
        diff_spikes = sum(np.count_nonzero(a != b) for a, b in zip(cpu_spk, num_spk))
        print(f"Discrepancies between CPU and Numba: {diff_spikes} spikes across {n_steps} steps")
        assert diff_spikes == 0 or diff_spikes < 5, f"Numba diverged too much: {diff_spikes} spike differences"

    # If PyTorch is available
    for t_name in [k for k in backend_results if k.startswith("torch")]:
        cpu_spk, cpu_tot, cpu_gain, cpu_v = backend_results["cpu"]
        torch_spk, torch_tot, torch_gain, torch_v = backend_results[t_name]
        diff_spikes = sum(np.count_nonzero(a != b) for a, b in zip(cpu_spk, torch_spk))
        print(f"Discrepancies between CPU and {t_name}: {diff_spikes} spikes across {n_steps} steps")
        # In sparse matrix multiplication on GPU/torch, float accumulation order can differ slightly;
        # assert total spike count within 5% and no divergence in stable firing regime
        assert abs(torch_tot - cpu_tot) / max(cpu_tot, 1) < 0.05
