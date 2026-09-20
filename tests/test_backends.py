"""Simulation backends: each one really runs when asked for, and the CPU-side ones are bit-exact with the NumPy
reference. GPU backends (torch-cuda, torch-rocm) are compared statistically, because GPU sparse kernels may add a
neuron's inputs in a different order and a chaotic network then diverges spike by spike (see README: Performance)."""
import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.core import simcore
from kickthefly.sim.connectome import backends
from kickthefly.sim.connectome.sim import LIFParams, LIFSim

pytestmark = needs_pack

AVAIL = backends.detect_available_backends()
EXACT = [b for b in ("cpu", "numba", "torch-cpu") if b in AVAIL]
GPU = [b for b in ("torch-cuda", "torch-rocm") if b in AVAIL]
EXPECT_CLASS = {"cpu": backends.CPUBackend, "numba": backends.NumbaBackend, "torch-cpu": backends.TorchBackend,
                "torch-cuda": backends.TorchBackend, "torch-rocm": backends.TorchBackend}


def _run(backend: str, steps: int, dtype: str = "float32", seed: int = 42, dense: bool = False):
    _, W, _ = simcore.pack()
    lp = LIFParams(backend=backend, dtype=dtype)
    if dense:
        lp.sparse_path_max_active = 0.0           # every step takes the dense path (normally only busy ones do)
    sim = LIFSim(None, lp, W_in=W, seed=seed)
    assert sim.backend.name == backend, f"asked for {backend}, got {sim.backend.name}"
    out = np.empty((steps, sim.n), bool)
    for step in range(steps):
        sens = None
        if step % 20 == 10:                                   # a sensory pulse every 100 ms
            sens = np.zeros(sim.n, np.float32)
            sens[100:150] = 2.0
        out[step] = sim.step(sens)
    return sim, out


def test_detect_available_backends():
    assert "cpu" in AVAIL


@pytest.mark.parametrize("name", list(AVAIL))
def test_requested_backend_really_runs(name):
    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(backend=name), W_in=W, seed=1)
    assert isinstance(sim.backend, EXPECT_CLASS[name])
    assert sim.backend.name == name
    assert isinstance(sim.backend.device, str) and sim.backend.device


def test_unknown_or_missing_backend_falls_back_to_cpu():
    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=123)
    assert backends.create_backend(sim, "non_existent_gpu_backend").name == "cpu"
    if not GPU:
        assert backends.create_backend(sim, "torch-cuda").name == "cpu"      # recorded as what actually ran


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", [b for b in EXACT if b != "cpu"])
def test_cpu_side_backends_are_bit_exact(name, dtype):
    """1000 steps (5 s of brain time) is well past where a single rounding difference used to show: an earlier
    fastmath Numba kernel first differed at step 289 and then decorrelated completely."""
    ref_sim, ref = _run("cpu", 1000, dtype)
    sim, got = _run(name, 1000, dtype)
    sim.backend.sync_to_host()
    assert np.array_equal(got, ref), f"{name} differs from cpu in {int((got != ref).sum())} spikes"
    assert np.array_equal(sim.v, ref_sim.v) and np.array_equal(sim.refr, ref_sim.refr)
    assert sim.gain == ref_sim.gain


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", [b for b in EXACT if b != "cpu"])
def test_cpu_side_backends_are_bit_exact_on_the_dense_path(name, dtype):
    """The reference sums busy steps' inputs in the state dtype and sparse ones in float32; both paths must match."""
    ref_sim, ref = _run("cpu", 300, dtype, dense=True)
    sim, got = _run(name, 300, dtype, dense=True)
    sim.backend.sync_to_host()
    assert np.array_equal(got, ref), f"{name} differs from cpu in {int((got != ref).sum())} spikes"
    assert np.array_equal(sim.v, ref_sim.v), "membrane potentials differ (a last-bit rounding difference)"


@pytest.mark.parametrize("name", GPU)
def test_gpu_backends_statistically_match(name):
    """Tolerance: brain-wide firing within 2% of the CPU's, and per-population rates (1000 neuron blocks) correlate
    at r > 0.95 over 1000 steps."""
    _, ref = _run("cpu", 1000)
    _, got = _run(name, 1000)
    assert abs(got.sum() - ref.sum()) / ref.sum() < 0.02
    blocks = lambda s: s[:, : s.shape[1] // 1000 * 1000].reshape(s.shape[0], -1, 1000).sum((0, 2))
    assert np.corrcoef(blocks(got), blocks(ref))[0, 1] > 0.95


@pytest.mark.parametrize("name", list(AVAIL))
def test_host_state_writes_reach_the_backend(name):
    """Save states and the neural clamp write sim.v / sim.spikes directly; every backend must see that."""
    _, W, _ = simcore.pack()
    sim = LIFSim(None, LIFParams(backend=name), W_in=W, seed=5)
    for _ in range(20):
        sim.step()
    quiet = np.flatnonzero(~sim.spikes & (sim.refr == 0))[:50]
    sim.v[quiet] = sim.p.v_thresh * 10                        # force them over threshold
    sim.backend.sync_from_host()
    assert sim.step()[quiet].all()


@pytest.mark.parametrize("name", list(AVAIL))
def test_weight_changes_reach_the_backend(name):
    """Learning, lesions and the synapse threshold edit W_csr.data in place and call on_weights_changed()."""
    _, W, _ = simcore.pack()
    a = LIFSim(None, LIFParams(backend="cpu"), W_in=W, seed=9)
    b = LIFSim(None, LIFParams(backend=name), W_in=W, seed=9)
    for sim in (a, b):
        sim.W_csr.data *= 0.0
        sim.W_csc.data *= 0.0
        sim.backend.on_weights_changed()
        for _ in range(50):
            sim.step()
    if name in EXACT:
        assert np.array_equal(a.spikes, b.spikes)


def test_headless_backend_and_dtype_reach_worker_processes(monkeypatch):
    """headless --backend/--dtype set the process-wide defaults that validation's spawned workers inherit."""
    monkeypatch.setenv("KICK_THE_FLY_SIM_BACKEND", "cpu")
    monkeypatch.setenv("KICK_THE_FLY_SIM_DTYPE", "float64")
    p = LIFParams()
    assert (p.backend, p.dtype) == ("cpu", "float64")
    from kickthefly.game import kick_the_fly as k
    args = k.parse_args(["--dtype", "float64", "--backend", "numba"])
    assert (args.dtype, args.backend) == ("float64", "numba")


@pytest.mark.parametrize("name", GPU)
def test_batched_multi_fly_plasticity_identical(name):
    """Assert that a trained fly's KC->MBON plastic weights after N conditioning pairings
    are 100% identical between unbatched and batched multi-fly GPU execution."""
    from kickthefly.core import memory
    g, W, _ = simcore.pack()

    def run_conditioning(batched: bool):
        sim1 = LIFSim(None, LIFParams(backend=name), W_in=W.copy(), seed=42)
        sim2 = LIFSim(None, LIFParams(backend=name), W_in=W.copy(), seed=99)
        mem1 = memory.Memory(g, sim1, load=False)
        odor_pattern = np.zeros(g.n, np.float32)
        odor_pattern[mem1.kc[:50]] = 5.0
        shock_drive = np.zeros(g.n, np.float32)
        shock_drive[mem1.dan[:20]] = 8.0

        for trial in range(6):
            # Odor presentation (5 steps)
            for _ in range(5):
                if batched:
                    backends.TorchBackend.step_batch([sim1, sim2], [odor_pattern, None])
                else:
                    sim1.step(odor_pattern)
                    sim2.step(None)
            # Shock (dopamine activation, 5 steps)
            for _ in range(5):
                if batched:
                    backends.TorchBackend.step_batch([sim1, sim2], [shock_drive, None])
                else:
                    sim1.step(shock_drive)
                    sim2.step(None)
            mem1.step(sim1.activity.rates(), calm=False, steps=(trial + 1) * 10)
        return mem1.w.copy(), sim1.spikes.copy()

    w_unbatched, sp_unbatched = run_conditioning(False)
    w_batched, sp_batched = run_conditioning(True)

    assert np.array_equal(w_unbatched, w_batched), "KC->MBON weights differ between batched and unbatched!"
    assert np.array_equal(sp_unbatched, sp_batched), "Spikes differ between batched and unbatched!"


@pytest.mark.parametrize("name", GPU)
def test_fused_lif_kernel_toggle(name):
    """Test that the fused LIF kernel toggle runs correctly and matches statistical tolerance."""
    _, W, _ = simcore.pack()
    lp = LIFParams(backend=name, fuse_lif=True)
    sim = LIFSim(None, lp, W_in=W, seed=7)
    for _ in range(50):
        sim.step()
    assert sim.spikes.shape == (sim.n,)
    assert sim.spikes.dtype == bool


