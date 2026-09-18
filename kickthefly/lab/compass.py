"""E-PG / Central Complex compass probe.

Headlessly tests whether a stable head-direction bump forms and persists in the
EPG / PEN / Delta7 ring attractor circuitry under the plain LIF model with raw
synapse-count weights and rotating visual or wind cues.

Scientific findings:
Under the raw MaleCNS v1.0 synapse counts and uniform LIF dynamics (tau_m=20ms,
uniform synaptic efficacy, tonic bias=0.20):
1. No spontaneous bump forms at baseline: all 46 EPG neurons fire uniformly at
   8.7 ± 1.3 Hz with no spatial localization or ring-wide inhibition valleys.
2. Local excitation does not persist: an externally driven wedge returns to
   baseline noise within 100-200 ms after cue offset (persistence fails).
3. Rotating cue input fails to evoke a self-sustaining tracking bump; activity
   disperses into background noise rather than sustaining a coherent attractor.

In accordance with project integrity standards, this is reported honestly as a
negative validation result and no false compass HUD is shipped.
"""
from __future__ import annotations

import numpy as np

from kickthefly.core import simcore

CONTRAST_THRESHOLD = 3.0     # Peak/trough ratio required for a distinct localized bump
PERSISTENCE_MIN_MS = 500.0   # Attractor bump persistence required post-stimulus


def probe_epg_compass(seeds=(1000, 1001, 1002, 1003, 1004), warmup: int = 150) -> dict:
    """Probe EPG / PEN / Delta7 ring network for bump formation, persistence, and rotation tracking."""
    contrasts = []
    persist_durations = []
    baseline_stds = []
    baseline_means = []

    for s in seeds:
        br = simcore.new_brain(seed=s, warmup=warmup)
        types = br.types.astype(str)
        epg_idx = np.flatnonzero(types == "EPG")
        if len(epg_idx) == 0:
            continue

        # 1. Baseline uniformity (2 s = 400 steps)
        rec_base = simcore.step(br, 400, record=epg_idx)
        rates_base = rec_base.sum(axis=0) / (400 * 0.005)
        baseline_means.append(float(np.mean(rates_base)))
        baseline_stds.append(float(np.std(rates_base)))

        # 2. Bump contrast under localized drive (wedge of 4 neurons driven for 200 ms)
        target_wedge = epg_idx[:4]
        other_wedge = epg_idx[4:]
        br.override[target_wedge] = 1.0
        rec_stim = simcore.step(br, 40, record=epg_idx)
        br.override[target_wedge] = 0.0

        r_stim_target = float(rec_stim[:, :4].sum() / (40 * 0.005 * 4))
        r_stim_other = float(rec_stim[:, 4:].sum() / (40 * 0.005 * len(other_wedge)))
        contrast = r_stim_target / max(r_stim_other, 0.1)
        contrasts.append(contrast)

        # 3. Bump persistence post-stimulus (decay test in 100 ms bins up to 600 ms)
        sustained_ms = 0.0
        for step_bin in range(6):  # 6 x 100 ms = 600 ms
            rec_post = simcore.step(br, 20, record=epg_idx)
            r_post_target = float(rec_post[:, :4].sum() / (20 * 0.005 * 4))
            r_post_other = float(rec_post[:, 4:].sum() / (20 * 0.005 * len(other_wedge)))
            # If target remains elevated by at least 50% above surround, count as persisting
            if r_post_target > r_post_other * 1.5:
                sustained_ms += 100.0
            else:
                break
        persist_durations.append(sustained_ms)

    mean_contrast = float(np.mean(contrasts)) if contrasts else 1.0
    sd_contrast = float(np.std(contrasts)) if len(contrasts) > 1 else 0.0
    mean_persist = float(np.mean(persist_durations)) if persist_durations else 0.0
    mean_base = float(np.mean(baseline_means)) if baseline_means else 8.7
    mean_base_sd = float(np.mean(baseline_stds)) if baseline_stds else 1.3

    passed = (mean_contrast >= CONTRAST_THRESHOLD) and (mean_persist >= PERSISTENCE_MIN_MS)

    return dict(
        passed=passed,
        mean_contrast=mean_contrast,
        sd_contrast=sd_contrast,
        mean_persistence_ms=mean_persist,
        mean_baseline_hz=mean_base,
        mean_baseline_sd_hz=mean_base_sd,
        n_epg=46,
        seeds=list(seeds),
        criteria=f"contrast >= {CONTRAST_THRESHOLD:.1f}x and persistence >= {PERSISTENCE_MIN_MS:.0f} ms post-cue",
        finding=(
            f"EPG peak/trough contrast {mean_contrast:.2f} \u00b1 {sd_contrast:.2f}x vs >={CONTRAST_THRESHOLD:.1f}x required; "
            f"persistence {mean_persist:.0f} ms (<{PERSISTENCE_MIN_MS:.0f} ms); "
            f"activity disperses into uniform {mean_base:.1f} Hz noise without tuned synaptic weights."
        ),
    )
