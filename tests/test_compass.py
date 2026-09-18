"""Tests for the E-PG compass bump formation and persistence probe.

Verifies that the probe accurately evaluates the central complex ring network
and honestly reports the negative result (no persistent bump under raw LIF weights).
"""
import pytest
from conftest import needs_pack

pytestmark = [needs_pack]


def test_epg_compass_negative_validation():
    from kickthefly.lab import compass

    # Run with 2 seeds for fast test execution
    res = compass.probe_epg_compass(seeds=(1000, 1001), warmup=50)

    assert "passed" in res
    assert res["passed"] is False  # Must fail honestly without weight tuning
    assert res["n_epg"] == 46
    assert res["mean_contrast"] < compass.CONTRAST_THRESHOLD
    assert res["mean_persistence_ms"] < compass.PERSISTENCE_MIN_MS
    assert "finding" in res
    assert "noise without tuned synaptic weights" in res["finding"]
