"""Validation: does this connectome sim reproduce published fly results? Headless, with numbers, pass or fail.

    python kick_the_fly.py --validate [--out results.json] [--workers N]
    pytest tests/test_validation.py

Every cell type named here was checked against the MaleCNS v1.0 annotations (the synonyms column records the
published names: DNg62 and DNge078 are "Hampel 2015: aDN1/aDN2", DNp01 is the giant fiber, MDN is the moonwalker
descending neuron, GNG540/GNG550 are "Yao & Scott 2022: Sugar SEL PN", DNg28 is "Yao & Scott 2022: Bitter-SEL").

Method (fixed before the held-out run; see README "Validation"):
  - Seeds 1000-1009, never used while developing the assays (exploratory probing used seeds 0-299).
  - Pathway tests: after 2 s of calm, a set of neurons is held driven for 2 s (current every step, like optogenetic
    activation) and a readout population's firing is compared with the 2 s before. The same brain state (snapshot) is
    also run with a control set of the same size. Pass: mean driven/baseline ratio >= 1.5 AND the ratio exceeds the
    control's in a one-sided Wilcoxon signed-rank test, p < 0.01, over the 10 paired seeds.
  - Conditioning: T-maze with reciprocal halves (assays.py). Pass: mean PI >= 0.5 AND |mean PI of the unpaired control|
    <= 0.25 AND paired > unpaired (one-sided Wilcoxon, p < 0.01).
  - The thresholds (1.5x, p < 0.01, PI 0.5) were chosen by us for this release after exploratory probing; they are not
    from the papers. A pass means the sim shows the effect in the direction and at the strength stated, not that its
    numbers match the papers'.

Behaviors that fail here are reported as failures in Lab mode and never get a "Real flies do this too" popup.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEEDS = tuple(range(1000, 1010))
RATIO_MIN, P_MAX, PI_MIN, CONTROL_PI_MAX = 1.5, 0.01, 0.5, 0.25
PRE, STIM = 400, 400
RESULTS_NAME = "validation_results.json"

TESTS = (
    dict(id="looming_escape", name="Looming detectors drive the giant fiber escape neuron",
         play="It dodged something rushing at it. Real flies escape looming objects through this same giant fiber.",
         claim="Activating the looming-sensitive visual neurons LPLC2 and LC4 excites the giant fiber DNp01.",
         citation="von Reyn et al. 2014, Nat Neurosci 17:962; Ache et al. 2019, Curr Biol 29:1073",
         drive="loom", drive_label="LPLC2 + LC4 (311)", readout="dnp01", readout_label="DNp01 (giant fiber)",
         control="vpn", control_label="311 random other visual projection neurons", popup_event="DODGE"),
    dict(id="mdn_backward", name="Moonwalker neurons drive backward walking",
         play="",
         claim="Activating MDN drives the leg motor program for walking backward.",
         citation="Bidaye et al. 2014, Science 344:97",
         drive="mdn", drive_label="MDN (4)", readout="mn_legs", readout_label="leg motor neurons (T1-T3, 381)",
         control="dn", control_label="4 random other descending neurons", popup_event=None,
         note="The sim has no walking rhythm; this asks whether MDN activity reaches the leg motor neurons at all."),
    dict(id="sugar_feeding", name="Sugar-pathway taste neurons reach the proboscis motor neuron",
         play="It reached for the sugar with its proboscis. Real flies' sugar-sensing neurons trigger this.",
         claim="Activating sugar-sensing gustatory neurons activates MN9, a motor neuron for proboscis extension; "
               "bitter-sensing neurons don't.",
         citation="Shiu et al. 2024, Nature 634:210 (sugar SEL PNs: Yao & Scott 2022)",
         drive="sweet", drive_label="30 sugar-pathway gustatory neurons", readout="mn9", readout_label="MN9",
         control="bitter", control_label="30 bitter-pathway gustatory neurons", popup_event="PROBOSCIS",
         note="GRNs aren't labeled by taste in this dataset: the sugar and bitter sets are chosen from wiring to the "
              "Yao & Scott 2022 sugar and bitter SEL neurons (assays.py). MN9 is not used to choose them."),
    dict(id="antenna_grooming_circuit", name="Antennal touch neurons excite the antennal grooming command neurons",
         play="Wind on its antennae excited the neurons that start antennal grooming. Real flies use these same "
              "neurons to clean their antennae.",
         claim="Antennal mechanosensory neurons (JO-C/E) excite the descending neurons aDN1/aDN2 of the antennal "
               "grooming circuit.",
         citation="Hampel et al. 2015, eLife 4:e08758; Shiu et al. 2024, Nature 634:210",
         drive="jo_ce", drive_label="JO-C/E mechanosensory neurons (335)", readout="adn",
         readout_label="aDN1/aDN2 (DNg62, DNge078)", control="sensory", control_label="335 random other sensory neurons",
         popup_event="GROOM"),
    dict(id="adn_grooming_motor", name="aDN activation drives front-leg (grooming) motor neurons",
         play="",
         claim="Activating aDN1/aDN2 drives antennal grooming, a front-leg movement.",
         citation="Hampel et al. 2015, eLife 4:e08758",
         drive="adn", drive_label="aDN1/aDN2 (4)", readout="mn_front", readout_label="front-leg motor neurons (135)",
         control="dn", control_label="4 random other descending neurons", popup_event=None,
         note="The sim has no leg movement; this asks whether aDN activity reaches the front-leg motor neurons."),
    dict(id="mb_conditioning", name="Mushroom body conditioning: T-maze performance index",
         play="It avoided the smell it learned to fear. Real flies learn smells in this same part of the brain.",
         claim="Pairing an odor with shock makes flies avoid it in a T-maze (positive PI); unpaired odor and shock "
               "don't.",
         citation="Tully & Quinn 1985, J Comp Physiol A 157:263",
         drive="", drive_label="odor + shock (6 cycles)", readout="", readout_label="T-maze choices",
         control="", control_label="same odors and shocks, unpaired", popup_event="AVOID",
         note="The plasticity rule, the shock's link to dopamine, and the choice at the T-maze are game rules on top of "
               "the connectome's real KC -> MBON synapses; this tests that they produce odor-specific memory."),
    dict(id="epg_compass", name="E-PG central complex compass forms an orientation bump",
         play="",
         claim="E-PG / PEN / Delta7 recurrent ring attractor forms a persistent head-direction bump.",
         citation="Seelig & Jayaraman 2015, Nature 521:186; Green et al. 2017, Nature 546:101",
         drive="", drive_label="localized wedge stimulation", readout="epg", readout_label="EPG (46)",
         control="", control_label="none", popup_event=None,
         note="Under raw unweighted LIF dynamics without tuned E/I balance, bump contrast and persistence fail; "
              "reported honestly as a negative validation result."),
)
BY_ID = {t["id"]: t for t in TESTS}
# The held-out results of this release (README "Validation"). tests/test_validation.py and --strict flag any change,
# in either direction, so a regression (or a newly reproduced behavior) never goes unnoticed.
EXPECTED = {"looming_escape": True, "mdn_backward": False, "sugar_feeding": True, "antenna_grooming_circuit": True,
            "adn_grooming_motor": False, "mb_conditioning": True, "epg_compass": False}


def _ratio(base: float, driven: float) -> float:
    return driven / max(base, 0.5)


def _pathway_seed(seed: int, wiring=None) -> dict:
    """All pathway tests for one seed, each drive and its control starting from the same brain snapshot."""
    from kickthefly.lab import assays
    from kickthefly.core import savestate
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=seed, wiring=wiring)
    g = assays.groups(br)
    snap: dict = {}
    meta = savestate.brain_state(br, "s_", snap)
    out = {}
    for t in TESTS:
        if not t["drive"]:
            continue
        drive = g[t["drive"]]
        if t["control"] in ("bitter",):
            control = g["bitter"]
        else:
            exclude = np.concatenate([drive, g[t["readout"]]])
            control = assays.random_like(g[t["control"]], len(drive), exclude, seed * 31 + len(out))
        res = {}
        for label, rows in (("drive", drive), ("control", control)):
            savestate.restore_brain(br, meta, snap, "s_")
            r = assays.pathway_response(br, rows, {"readout": g[t["readout"]]}, pre=PRE, stim=STIM)["readout"]
            res[label] = dict(base_hz=r[0], driven_hz=r[1], ratio=_ratio(*r))
        out[t["id"]] = res
    return out


def _tmaze_seed(args, wiring=None) -> dict:
    from kickthefly.lab import assays

    seed, cs_plus, paired = args
    return assays.tmaze_fly(seed if cs_plus == "odor_a" else seed + 50_000, cs_plus, paired=paired, wiring=wiring)


def _wilcoxon_greater(a, b) -> float:
    from scipy import stats

    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox").pvalue)


def run(seeds=SEEDS, workers: int | None = None, progress=None, include=None, wiring=None) -> dict:
    """Run the suite. progress(done, total, label) is called as work finishes.

    wiring: a sim.wiring.Wiring every fly is built with (the threshold sweep and the sign-flip stress test use this
    to ask which of these results survive a changed connectome). The pass criteria are the same either way.
    """
    from kickthefly.lab import assays
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import lab
    from kickthefly.core import simcore
    from kickthefly.core.version import __version__

    t0 = time.time()
    workers = workers or min(4, os.cpu_count() or 1)
    include = set(include or BY_ID)
    jobs_path = [s for s in seeds] if include - {"mb_conditioning", "epg_compass"} else []
    jobs_tmaze = [(s, cs, paired) for s in seeds for paired in (True, False) for cs in ("odor_a", "odor_b")] \
        if "mb_conditioning" in include else []
    total, done = len(jobs_path) + len(jobs_tmaze), 0
    path_res, tmaze_res = {}, []

    def tick(label):
        nonlocal done
        done += 1
        if progress:
            progress(done, total, label)

    if workers > 1:
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {ex.submit(_pathway_seed, s, wiring): ("path", s) for s in jobs_path}
            futs.update({ex.submit(_tmaze_seed, j, wiring): ("tmaze", j) for j in jobs_tmaze})
            from concurrent.futures import as_completed
            for f in as_completed(futs):
                kind, key = futs[f]
                if kind == "path":
                    path_res[key] = f.result()
                else:
                    tmaze_res.append((key, f.result()))
                tick(kind)
    else:
        for s in jobs_path:
            path_res[s] = _pathway_seed(s, wiring)
            tick("path")
        for j in jobs_tmaze:
            tmaze_res.append((j, _tmaze_seed(j, wiring)))
            tick("tmaze")

    results = []
    for t in TESTS:
        if t["id"] not in include:
            continue
        r = {k_: v for k_, v in t.items()}
        if t["id"] == "epg_compass":
            from kickthefly.lab import compass
            res_epg = compass.probe_epg_compass(seeds=seeds)
            r["measured"] = dict(
                mean_contrast=res_epg["mean_contrast"],
                sd_contrast=res_epg["sd_contrast"],
                mean_persistence_ms=res_epg["mean_persistence_ms"],
                mean_baseline_hz=res_epg["mean_baseline_hz"],
                mean_baseline_sd_hz=res_epg["mean_baseline_sd_hz"],
                n=len(seeds),
            )
            r["criteria"] = res_epg["criteria"]
            r["passed"] = res_epg["passed"]
            r["finding"] = res_epg["finding"]
        elif t["drive"]:
            per = [dict(seed=s, **path_res[s][t["id"]]) for s in seeds]
            dr = [p["drive"]["ratio"] for p in per]
            cr = [p["control"]["ratio"] for p in per]
            p = _wilcoxon_greater(dr, cr)
            r["measured"] = dict(
                drive_ratio_mean=float(np.mean(dr)), drive_ratio_sd=float(np.std(dr, ddof=1)),
                control_ratio_mean=float(np.mean(cr)), control_ratio_sd=float(np.std(cr, ddof=1)),
                readout_base_hz=float(np.mean([x["drive"]["base_hz"] for x in per])),
                readout_driven_hz=float(np.mean([x["drive"]["driven_hz"] for x in per])),
                p_value=p, n=len(per))
            r["criteria"] = f"drive ratio >= {RATIO_MIN} and > control (one-sided Wilcoxon p < {P_MAX})"
            r["passed"] = bool(np.mean(dr) >= RATIO_MIN and p < P_MAX)
            r["per_seed"] = per
        else:
            by = {}
            for (s, cs, paired), res in tmaze_res:
                by.setdefault((s, paired), []).append(res)
            pi_p = [float(np.mean([h["pi"] for h in by[(s, True)]])) for s in seeds]
            pi_u = [float(np.mean([h["pi"] for h in by[(s, False)]])) for s in seeds]
            p = _wilcoxon_greater(pi_p, pi_u)
            fear_p = [float(np.mean([h["fear_plus"] for h in by[(s, True)]])) for s in seeds]
            fear_m = [float(np.mean([h["fear_minus"] for h in by[(s, True)]])) for s in seeds]
            mb_p = [float(np.mean([h["mbon_plus_hz"] for h in by[(s, True)]])) for s in seeds]
            mb_m = [float(np.mean([h["mbon_minus_hz"] for h in by[(s, True)]])) for s in seeds]
            r["measured"] = dict(pi_mean=float(np.mean(pi_p)), pi_sd=float(np.std(pi_p, ddof=1)),
                                 control_pi_mean=float(np.mean(pi_u)), control_pi_sd=float(np.std(pi_u, ddof=1)),
                                 fear_cs_plus=float(np.mean(fear_p)), fear_cs_minus=float(np.mean(fear_m)),
                                 approach_mbon_cs_plus_hz=float(np.mean(mb_p)),
                                 approach_mbon_cs_minus_hz=float(np.mean(mb_m)), p_value=p, n=len(seeds))
            r["criteria"] = (f"PI >= {PI_MIN}, |unpaired PI| <= {CONTROL_PI_MAX}, paired > unpaired "
                             f"(one-sided Wilcoxon p < {P_MAX})")
            r["passed"] = bool(np.mean(pi_p) >= PI_MIN and abs(np.mean(pi_u)) <= CONTROL_PI_MAX and p < P_MAX)
            r["per_seed"] = [dict(seed=s, pi=a, control_pi=b) for s, a, b in zip(seeds, pi_p, pi_u)]
        results.append(r)

    g, W, _ = simcore.pack()
    return dict(app_version=__version__, created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
                seeds=list(seeds), workers=workers, n_neurons=int(g.n), synapses=int(W.nnz),
                wiring=(wiring.as_dict() if wiring is not None else None),
                lab_params=dict(lab.DEFAULTS), thresholds=dict(k.THRESH), loom=[k.LOOM_MIN, k.LOOM_FULL],
                sweet_n=assays.SWEET_N, tests=results)


def summary(res: dict) -> str:
    lines = [f"Kick the Fly {res['app_version']} validation, seeds {res['seeds'][0]}-{res['seeds'][-1]} "
             f"(n={len(res['seeds'])}), {res['seconds']}s"]
    for t in res["tests"]:
        m = t["measured"]
        if "drive_ratio_mean" in m:
            nums = (f"{t['readout_label']}: x{m['drive_ratio_mean']:.2f} when driving {t['drive_label']} vs "
                    f"x{m['control_ratio_mean']:.2f} for {t['control_label']}, p={m['p_value']:.4f}")
        elif "pi_mean" in m:
            nums = f"PI {m['pi_mean']:.2f} ± {m['pi_sd']:.2f} vs unpaired {m['control_pi_mean']:.2f}, p={m['p_value']:.4f}"
        elif "mean_contrast" in m:
            nums = f"contrast {m['mean_contrast']:.2f} ± {m.get('sd_contrast', 0):.2f}x, persist {m['mean_persistence_ms']:.0f} ms"
        else:
            nums = str(m)
        lines.append(f"  [{'PASS' if t['passed'] else 'FAIL'}] {t['name']}: {nums}")
    return "\n".join(lines)


# --- results for the game ---------------------------------------------------------------------------------------------
def bundled_path() -> Path | None:
    """validation_results.json shipped inside the exe/AppImage (produced by the build from the same pack)."""
    import sys

    import kickthefly

    for root in (Path(getattr(sys, "_MEIPASS", "")), kickthefly.DATA_DIR):
        if str(root) and (root / RESULTS_NAME).exists():
            return root / RESULTS_NAME
    return None


def local_path() -> Path:
    from kickthefly.core import paths

    return paths.get().data_dir / RESULTS_NAME


def load_results() -> tuple[dict | None, str]:
    """(results, source): a run on this PC wins over the build's bundled results."""
    for path, source in ((local_path(), "run on this PC"), (bundled_path(), "from this build")):
        if path is None or not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8")), source
        except (OSError, json.JSONDecodeError):
            continue
    return None, ""


def save_results(res: dict, path: Path | None = None) -> Path:
    path = path or local_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o),
                    encoding="utf-8")
    return path


def passing_events(res: dict | None, n_neurons: int, synapses: int) -> dict[str, dict]:
    """popup event -> test, for tests that passed on this brain pack with default parameters."""
    if not res or res.get("n_neurons") != n_neurons or res.get("synapses") != synapses:
        return {}
    from kickthefly.lab import lab

    if any(abs(float(res.get("lab_params", {}).get(k, v)) - v) > 1e-9 for k, v in lab.DEFAULTS.items()):
        return {}
    return {t["popup_event"]: t for t in res.get("tests", []) if t.get("passed") and t.get("popup_event")}
