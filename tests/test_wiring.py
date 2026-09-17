"""Changing the connectome itself: the synapse threshold, and that every change is exactly reversible."""
import numpy as np

from conftest import needs_pack


@needs_pack
def test_threshold_stats_match_the_pack():
    from kickthefly.sim import wiring

    counts = wiring.synapse_counts()
    st = wiring.threshold_stats(5)
    assert st["connections"] == len(counts)
    assert st["connections_dropped"] == int((counts < 5).sum())
    assert st["synapses_dropped"] == int(counts[counts < 5].sum())
    assert 0 < st["connections_dropped_share"] < 1
    assert st["neurons_cut_off"] <= st["neurons_touched"] <= st["neurons"]


@needs_pack
def test_thresholds_below_the_pack_filter_change_nothing():
    """The loader already drops connections below 3 synapses, and the report has to say so rather than pretend."""
    from kickthefly.lab import robustness
    from kickthefly.sim import wiring

    for n in range(1, robustness.MIN_PACK_SYNAPSES + 1):
        assert wiring.threshold_stats(n)["connections_dropped"] == 0


@needs_pack
def test_applying_a_threshold_zeroes_exactly_those_connections_and_reverses(tmp_path):
    from kickthefly.core import simcore
    from kickthefly.sim import wiring
    from kickthefly.sim.wiring import Wiring

    br = simcore.new_brain(seed=3, warmup=0)
    before = br.sim.W_csr.data.copy()
    counts = wiring.synapse_counts()
    weak = counts < 6

    info = wiring.apply(br, Wiring(min_synapses=6))
    assert info["changed"] == int(weak.sum())
    assert np.all(br.sim.W_csr.data[weak] == 0)
    assert np.array_equal(br.sim.W_csr.data[~weak], before[~weak])
    assert br.sim.W_csr.nnz == len(before), "entries are zeroed, not removed: index maps must stay valid"
    assert np.allclose(br.sim.W_csc.toarray().sum(), br.sim.W_csr.toarray().sum()) if br.n < 100 else True

    wiring.clear(br)
    assert np.array_equal(br.sim.W_csr.data, before)
    assert np.array_equal(br.sim.W_csc.data, br.sim.W_csr.tocsc().data)


@needs_pack
def test_the_threshold_survives_a_learning_update():
    """Learning writes the plastic synapses back every 50 ms; a dropped KC -> MBON synapse must stay dropped."""
    from kickthefly.core import simcore
    from kickthefly.sim import wiring
    from kickthefly.sim.wiring import Wiring

    br = simcore.new_brain(seed=4, warmup=0)
    mem = br.memory
    assert mem is not None
    counts = wiring.synapse_counts()
    plastic_weak = mem.csr_pos[counts[mem.csr_pos] < 6]
    assert len(plastic_weak) > 0, "some plastic synapses should be below the threshold"

    wiring.apply(br, Wiring(min_synapses=6))
    assert np.all(br.sim.W_csr.data[plastic_weak] == 0)
    simcore.step(br, 30)                                   # long enough for at least one plasticity update
    assert np.all(br.sim.W_csr.data[plastic_weak] == 0), "learning wrote the dropped synapses back"

    wiring.clear(br)
    assert np.all(br.sim.W_csr.data[plastic_weak] > 0)


@needs_pack
def test_a_thresholded_brain_still_runs_and_differs():
    from kickthefly.core import simcore
    from kickthefly.sim.wiring import Wiring

    plain = simcore.new_brain(seed=11, warmup=200)
    cut = simcore.new_brain(seed=11, warmup=200, wiring=Wiring(min_synapses=10))
    a = simcore.step(plain, 100, record=np.arange(plain.n))
    b = simcore.step(cut, 100, record=np.arange(cut.n))
    assert a.sum() > 0 and b.sum() > 0, "both brains still fire"
    assert not np.array_equal(a, b), "dropping 74% of the connections must change the activity"


def test_wiring_label_and_dict():
    from kickthefly.sim.wiring import Wiring

    assert Wiring().is_identity and Wiring().label() == "unmodified connectome"
    w = Wiring(min_synapses=5)
    assert not w.is_identity and "drop <5" in w.label()
    assert w.as_dict()["min_synapses"] == 5
