"""How much do this simulation's results depend on the reconstruction being right?

Two manipulations, one question. Both re-run the validation suite's behaviors on a changed connectome and report
which of them still pass:

  threshold_sweep    drop every connection reconstructed with fewer than N synapses, for a range of N, and report
                     the threshold at which each validated behavior breaks. A behavior that survives a high
                     threshold does not rest on weak, possibly spurious contacts.
  signflip_trials    flip the sign of neurons whose neurotransmitter prediction the dataset is least sure about,
                     over repeated randomized trials, and report how often each behavior survives.

Both use the validation suite's own pass criteria (validation.py), unchanged, so "breaks" means the same thing here
as on the validation dashboard. The runs are lockstep and seeded, so the same sweep gives the same answer twice.

The connectome supplies the synapse counts and the neurotransmitter predictions with their confidences. Which
thresholds to try, how many flip trials to run and how many seeds to average over are this game's choices, and the
brain pack is already filtered to connections of at least 3 synapses by the loader, so a threshold of 1-3 changes
nothing at all.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from kickthefly.sim.wiring import Wiring

# The behaviors worth asking about: the four that pass validation on the unmodified connectome. The two that fail
# there are still run when asked for, but "breaks at" means nothing for a behavior that never worked.
DEFAULT_TESTS = ("looming_escape", "sugar_feeding", "antenna_grooming_circuit", "mb_conditioning")
DEFAULT_THRESHOLDS = (1, 4, 5, 6, 8, 10)
MIN_PACK_SYNAPSES = 3            # kickthefly/sim/connectome/loader.py MIN_WEIGHT


def _results_by_id(res: dict) -> dict:
    out = {}
    for t in res["tests"]:
        m = t["measured"]
        out[t["id"]] = dict(
            passed=bool(t["passed"]), name=t["name"],
            effect=float(m.get("drive_ratio_mean", m.get("pi_mean", 0.0))),
            control=float(m.get("control_ratio_mean", m.get("control_pi_mean", 0.0))),
            p_value=float(m.get("p_value", 1.0)),
            metric="drive/baseline ratio" if "drive_ratio_mean" in m else "performance index")
    return out


def run_behaviors(wiring: Wiring, seeds, include=DEFAULT_TESTS, workers: int | None = None, progress=None) -> dict:
    """The validation suite's behaviors, re-run on a changed connectome, with its own pass criteria."""
    from kickthefly.lab import validation

    res = validation.run(seeds=tuple(seeds), workers=workers, progress=progress, include=set(include), wiring=wiring)
    return _results_by_id(res)


def threshold_sweep(thresholds=DEFAULT_THRESHOLDS, seeds=None, include=DEFAULT_TESTS,
                    workers: int | None = None, progress=None) -> dict:
    """Run the behaviors at each minimum-synapse threshold and report where each one breaks."""
    from kickthefly.lab import validation
    from kickthefly.sim import wiring as wiring_mod

    # The validation seeds by default: the pass criteria are validation's, and a one-sided Wilcoxon over fewer than
    # seven seeds cannot reach p < 0.01 however large the effect is, so a short sweep would report false breaks.
    seeds = tuple(seeds) if seeds else validation.SEEDS
    thresholds = sorted({max(1, int(t)) for t in thresholds})
    t0 = time.time()
    steps = []
    for i, n in enumerate(thresholds):
        def step_progress(done, total, label, i=i, n=n):
            if progress:
                progress(i, len(thresholds), f"threshold {n}: {done}/{total}")

        stats = wiring_mod.threshold_stats(n)
        res = run_behaviors(Wiring(min_synapses=n), seeds, include, workers, step_progress)
        steps.append(dict(threshold=n, stats=stats, behaviors=res))
    breaks = {}
    for test_id in include:
        passed_at = [s["threshold"] for s in steps if s["behaviors"].get(test_id, {}).get("passed")]
        failed_at = [s["threshold"] for s in steps if test_id in s["behaviors"]
                     and not s["behaviors"][test_id]["passed"]]
        first_fail = None
        for s in steps:                                  # the lowest threshold at which it fails and never recovers
            if test_id in s["behaviors"] and not s["behaviors"][test_id]["passed"]:
                later = [x for x in steps if x["threshold"] >= s["threshold"] and test_id in x["behaviors"]]
                if all(not x["behaviors"][test_id]["passed"] for x in later):
                    first_fail = s["threshold"]
                    break
        breaks[test_id] = dict(breaks_at=first_fail, survived=passed_at, failed=failed_at,
                               name=steps[0]["behaviors"].get(test_id, {}).get("name", test_id) if steps else test_id)
    return dict(kind="threshold_sweep", created=time.strftime("%Y-%m-%d %H:%M:%S"),
                seconds=round(time.time() - t0, 1), seeds=list(seeds), thresholds=thresholds,
                pack_min_synapses=MIN_PACK_SYNAPSES, steps=steps, breaks=breaks)


def sweep_csv(res: dict, path: Path) -> Path:
    """One row per behavior and threshold: what the effect was and whether it still passed."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["behavior", "threshold", "connections_dropped", "connections_dropped_share", "neurons_cut_off",
                    "metric", "effect", "control", "p_value", "passed"])
        for step in res["steps"]:
            st = step["stats"]
            for test_id, b in step["behaviors"].items():
                w.writerow([test_id, step["threshold"], st["connections_dropped"],
                            f"{st['connections_dropped_share']:.4f}", st["neurons_cut_off"], b["metric"],
                            f"{b['effect']:.4f}", f"{b['control']:.4f}", f"{b['p_value']:.5f}",
                            "pass" if b["passed"] else "fail"])
    return path


def summary(res: dict) -> str:
    lines = [f"Minimum-synapse threshold sweep, seeds {res['seeds']}, {res['seconds']}s",
             f"(the brain pack already drops connections below {res['pack_min_synapses']} synapses, so thresholds "
             f"of 1-{res['pack_min_synapses']} change nothing)"]
    for test_id, b in res["breaks"].items():
        where = f"breaks at >= {b['breaks_at']} synapses" if b["breaks_at"] else "survives every threshold tried"
        lines.append(f"  {b['name']}: {where}")
    return "\n".join(lines)


def save(res: dict, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{res['kind']}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    if res["kind"] == "threshold_sweep":
        sweep_csv(res, folder / "threshold_sweep.csv")
    return folder
