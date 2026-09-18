"""Tests for the targeted optogenetics laser module."""
import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.lab.laser import LaserState, BASE_STIM_CURRENT, BASE_SILENCE_CURRENT


def test_laser_state_defaults_and_toggles():
    ls = LaserState()
    assert ls.target_type == "dnp01"
    assert ls.mode == "activate"
    assert ls.intensity == 1.0
    assert ls.trigger_mode == "hold"
    assert ls.current_value() == pytest.approx(BASE_STIM_CURRENT)

    # Toggle mode to silence
    ls.toggle_mode()
    assert ls.mode == "silence"
    assert ls.current_value() == pytest.approx(BASE_SILENCE_CURRENT)

    # Intensity scaling
    ls.set_intensity(2.0)
    assert ls.current_value() == pytest.approx(BASE_SILENCE_CURRENT * 2.0)

    # Intensity clipping
    ls.set_intensity(5.0)
    assert ls.intensity == 3.0
    ls.set_intensity(0.01)
    assert ls.intensity == 0.1


def test_laser_trigger_hold_vs_pulse():
    ls = LaserState(trigger_mode="hold")
    assert not ls.is_active(now=1.0)
    ls.trigger_press(now=1.0)
    assert ls.is_active(now=1.0)
    assert ls.is_active(now=2.0)
    ls.trigger_release()
    assert not ls.is_active(now=2.1)

    # Pulse mode
    ls.set_trigger_mode("pulse")
    ls.pulse_duration = 0.20
    assert not ls.is_active(now=3.0)
    ls.trigger_press(now=3.0)
    assert ls.is_active(now=3.10)
    assert ls.is_active(now=3.19)
    assert not ls.is_active(now=3.25)


@needs_pack
def test_laser_resolve_target_and_apply():
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=100)
    ls = LaserState(target_type="dnp01", mode="activate", intensity=1.5)

    rows = ls.resolve_target_rows(br)
    assert len(rows) > 0, "dnp01 should resolve to at least 1 neuron"

    # Test applying when not firing or not hitting
    cur = ls.apply(br, now=1.0, is_hitting=False)
    assert cur == 0.0
    assert np.all(br.override[rows] == 0.0)

    # Press trigger and hit fly
    ls.trigger_press(now=1.0)
    cur = ls.apply(br, now=1.0, is_hitting=True)
    expected_cur = BASE_STIM_CURRENT * 1.5
    assert cur == pytest.approx(expected_cur)
    assert np.all(br.override[rows] == pytest.approx(expected_cur))
    assert br.surgery is True

    # Move beam off fly: override cleared
    cur = ls.apply(br, now=1.05, is_hitting=False)
    assert cur == 0.0
    assert np.all(br.override[rows] == 0.0)
    assert br.surgery is False

    # Switch to silence and hit fly
    ls.set_mode("silence")
    ls.set_intensity(1.0)
    cur = ls.apply(br, now=1.1, is_hitting=True)
    assert cur == pytest.approx(BASE_SILENCE_CURRENT)
    assert np.all(br.override[rows] == pytest.approx(BASE_SILENCE_CURRENT))

    # Clear completely
    ls.clear(br)
    assert np.all(br.override[rows] == 0.0)
    assert not ls.is_active(now=1.1)
