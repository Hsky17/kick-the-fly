"""Assays: T-maze olfactory conditioning, looming escape, and sugar response, run headless in lockstep.

The in-game challenges (challenges.py), Lab results, repeated trials, protocols and the validation suite all use
these functions. Every run steps brains synchronously (simcore.py), so the same seed gives the same result.

What comes from the connectome and what is a game rule, per assay:

T-maze (Tully & Quinn 1985)
  Connectome: the odors reach Kenyon cells through the real olfactory pathway; the plastic synapses are the
    connectome's 41,495 KC -> MBON synapses, gated by its own PPL1/PAM dopamine -> MBON wiring (memory.py).
  Game rules: each odor is a fixed set of 6 olfactory glomeruli (not a real chemical's receptor profile); the shock
    drives PPL1 dopamine neurons and leg touch neurons directly; the plasticity rule and its rate (memory.py); and the
    choice. The sim has no T-maze body, so at the choice point the fly smells each arm and picks the one whose learned
    approach drive (liking minus fear, read from the synapses) is higher, plus decision noise. The standard metric is
    computed exactly as in the paper: PI = (flies choosing CS- minus flies choosing CS+) / all choices, averaged over
    reciprocal halves (each odor serving once as CS+). The control is explicitly unpaired: the same odors and the same
    shocks, but the shocks fall in the rest periods.

Looming escape (von Reyn et al. 2014; Ache et al. 2019)
  Connectome: LPLC2/LC4 -> giant fiber DNp01 and everything the sim does in between.
  Game rules: how an approaching object becomes LPLC2/LC4 drive (angular expansion speed above LOOM_MIN, full at
    LOOM_MIN + LOOM_FULL; the same code path as the game), and escaping when DNp01 exceeds THRESH["escape"] x calm.
    Because of that transduction threshold, part of the speed dependence is a game rule, not a measurement.

Sugar response (Shiu et al. 2024; sugar SEL projection neurons from Yao & Scott 2022)
  Connectome: which neurons are "sugar pathway" GRNs, and MN9's response. The dataset doesn't label GRNs by taste
    quality, so the sugar set is defined from wiring: the 30 head gustatory neurons (LB*, PhG*, claw_tpGRN,
    dorsal_tpGRN) with the most one- and two-synapse input to the sugar SEL PNs (GNG540, GNG550, annotated "Yao & Scott
    2022: Sugar SEL PN") minus their input to the bitter SEL neuron DNg28 ("Yao & Scott 2022: Bitter-SEL"). The bitter
    set is the 30 with the opposite preference. MN9 (a proboscis motor neuron) is never used to choose them.
  Game rules: dose is the share of sugar-pathway GRNs driven; a "proboscis extension" is MN9 firing at least
    PER_RATIO x its rate in the second before.
"""
from __future__ import annotations

import math

import numpy as np

import simcore

STEPS_PER_TICK = (4, 3, 3)          # 10 brain steps (50 ms) per 3 game ticks (1/20 s)
SWEET_N = 30
PER_RATIO = 1.5
ODOR_GLOMERULI_SEED = 1985           # which 6 glomeruli each T-maze odor uses (game rule)


# --- neuron groups -------------------------------------------------------------------------------------------------
def groups(br) -> dict[str, np.ndarray]:
    """Named neuron sets for assays and validation, computed once per brain from the pack's labels and wiring."""
    cached = getattr(br, "_assay_groups", None)
    if cached is not None:
        return cached
    t, sc = br.types, br.superclass
    sub = getattr(br, "subclass", np.full(br.n, "", dtype="<U1"))

    def T(*names):
        return np.flatnonzero(np.isin(t, names))

    motor = sc == "vnc_motor"
    head_gust = np.flatnonzero((np.char.startswith(t, "LB") | np.char.startswith(t, "PhG") |
                                np.char.startswith(t, "claw_tpGRN") | np.char.startswith(t, "dorsal_tpGRN"))
                               & (sc == "cb_sensory"))
    g = dict(
        dnp01=T("DNp01"), loom=np.concatenate([T("LPLC2"), T("LC4")]), mdn=T("MDN"), dnp09=T("DNp09"),
        adn=T("DNg62", "DNge078"),                       # "Hampel 2015: aDN1" / "aDN2" in the dataset's synonyms
        jo_ce=np.flatnonzero((np.char.startswith(t, "JO-C") | np.char.startswith(t, "JO-E")) & (np.char.find(sc, "sensory") >= 0)),
        mn9=T("MN9"), sugar_pn=T("GNG540", "GNG550"), bitter_sel=T("DNg28"),
        mn_front=np.flatnonzero(motor & (sub == "fl")), mn_mid=np.flatnonzero(motor & (sub == "ml")),
        mn_hind=np.flatnonzero(motor & (sub == "hl")), head_gust=head_gust,
        dn=np.flatnonzero(sc == "descending_neuron"), sensory=np.flatnonzero(np.char.find(sc, "sensory") >= 0),
        vpn=np.flatnonzero(sc == "visual_projection"),
    )
    g["mn_legs"] = np.concatenate([g["mn_front"], g["mn_mid"], g["mn_hind"]])
    W = abs(br.sim.W_csr)

    def reach(targets):                               # one- plus two-synapse input from each head GRN onto targets
        into = np.asarray(W[targets].sum(axis=0)).ravel()
        one = into[head_gust]
        two = np.asarray(W[:, head_gust].T @ into).ravel()
        return one + two

    pref = reach(g["sugar_pn"]) - reach(g["bitter_sel"])
    order = np.argsort(-pref, kind="stable")
    g["sweet"], g["bitter"] = head_gust[order[:SWEET_N]], head_gust[order[-SWEET_N:]]
    br._assay_groups = g
    return g


def random_like(pool: np.ndarray, n: int, exclude: np.ndarray, seed: int) -> np.ndarray:
    """A random control set of n neurons from pool, never overlapping exclude (seeded, so reproducible)."""
    cand = np.setdiff1d(pool, exclude)
    return np.sort(np.random.default_rng(seed).choice(cand, size=min(n, len(cand)), replace=False))


def hz(spike_counts: int, n_neurons: int, steps: int, dt: float = 0.005) -> float:
    return spike_counts / max(1, n_neurons) / max(1, steps) / dt


# --- pathway response (validation and Lab) ----------------------------------------------------------------------------
def pathway_response(br, drive_rows, readouts: dict[str, np.ndarray], pre: int = 400, stim: int = 400,
                     amp: float = 0.5) -> dict[str, tuple[float, float]]:
    """Hold drive_rows driven for `stim` steps after `pre` calm steps. Returns {readout: (baseline Hz, driven Hz)}.
    Driven neurons get current every step, the way optogenetic activation does in the experiments."""
    counts = {k: [0, 0] for k in readouts}
    for phase, n in ((0, pre), (1, stim)):
        if phase == 1 and drive_rows is not None and len(drive_rows):
            simcore.drive(br, drive_rows, amp)
        for _ in range(n):
            br._step()
            s = br.sim.spikes
            for k, rows in readouts.items():
                counts[k][phase] += int(np.count_nonzero(s[rows]))
    if drive_rows is not None and len(drive_rows):
        simcore.undrive(br, drive_rows)
    return {k: (hz(c[0], len(readouts[k]), pre), hz(c[1], len(readouts[k]), stim)) for k, c in counts.items()}


# --- T-maze olfactory conditioning ------------------------------------------------------------------------------------
def _tick_steps(i: int) -> int:
    return STEPS_PER_TICK[i % 3]


def present(br, odors, steps: int, observe: bool = True, shock: bool = False, record_mbon: list | None = None) -> None:
    """Deliver odors (and optionally shock) for `steps` brain steps, re-poking every tick like the game does."""
    mem = br.memory
    done, tick = 0, 0
    while done < steps:
        for o in odors:
            br.poke("scent", o, 0.5)
        if shock:
            br.poke("punish", None, 1.0)
            br.poke("legs", "L", 0.6)
            br.poke("legs", "R", 0.6)
        n = min(_tick_steps(tick), steps - done)
        for _ in range(n):
            br._step()
        done += n
        tick += 1
        if observe and mem is not None and tick % 3 == 0 and len(odors) == 1:
            r = br.sim.activity.rates()
            mem.observe(odors[0], r)
            if record_mbon is not None:
                record_mbon.append(mem.mbon_response(r))


def rest(br, steps: int) -> None:
    for _ in range(steps):
        br._step()


def tmaze_fly(seed: int, cs_plus: str = "odor_a", cycles: int = 6, test_trials: int = 20, paired: bool = True,
              decision_noise: float = 0.08, surgery: dict | None = None, params: dict | None = None,
              brain=None) -> dict:
    """One fly: train CS+ with shock (paired) or with shocks in the rests (unpaired control), then T-maze choices."""
    br = brain or simcore.new_brain(seed=seed, params=params)
    cs_minus = "odor_b" if cs_plus == "odor_a" else "odor_a"
    apply_surgery(br, surgery)
    rest(br, 500)                                    # memory.py learns resting activity before any plasticity
    for odor in (cs_plus, cs_minus):                 # naive exposure so each odor has a Kenyon cell template
        present(br, [odor], 240)
        rest(br, 160)
    naive_plus, naive_minus = br.memory.memory_of(cs_plus), br.memory.memory_of(cs_minus)
    for _ in range(cycles):
        present(br, [cs_plus], 240)                  # 1.2 s odor
        present(br, [cs_plus], 200, shock=paired)    # 1.0 s odor + shock (paired) or odor alone
        rest(br, 60)
        if not paired:
            present(br, [], 200, observe=False, shock=True)     # unpaired: the same shock, with no odor
        rest(br, 200)
        present(br, [cs_minus], 440)                 # CS- for as long, never with shock
        rest(br, 260)
    rest(br, 200)
    rng = np.random.default_rng(seed * 7919 + 17)
    mbon = {cs_plus: [], cs_minus: []}
    choices_plus = choices_minus = 0
    for trial in range(test_trials):
        first = (cs_plus, cs_minus) if trial % 2 == 0 else (cs_minus, cs_plus)
        drive = {}
        for odor in first:                           # the fly samples each arm's odor at the choice point
            present(br, [odor], 60, record_mbon=mbon[odor])
            fear, like = br.memory.memory_of(odor)
            drive[odor] = like - fear
            rest(br, 40)
        pick_plus = drive[cs_plus] + rng.normal(0, decision_noise) > drive[cs_minus] + rng.normal(0, decision_noise)
        choices_plus += int(pick_plus)
        choices_minus += int(not pick_plus)
        rest(br, 60)
    fear_plus, _ = br.memory.memory_of(cs_plus)
    fear_minus, _ = br.memory.memory_of(cs_minus)
    n = choices_plus + choices_minus
    return dict(seed=seed, cs_plus=cs_plus, paired=paired, cycles=cycles, pi=(choices_minus - choices_plus) / max(1, n),
                chose_plus=choices_plus, chose_minus=choices_minus, fear_plus=fear_plus, fear_minus=fear_minus,
                naive_fear_plus=naive_plus[0], naive_fear_minus=naive_minus[0],
                mbon_plus_hz=float(np.mean(mbon[cs_plus])) if mbon[cs_plus] else float("nan"),
                mbon_minus_hz=float(np.mean(mbon[cs_minus])) if mbon[cs_minus] else float("nan"))


def tmaze_group(seeds, paired: bool = True, **kw) -> dict:
    """Reciprocal design: half the flies learn odor A as CS+, half odor B. PI per reciprocal pair and overall."""
    flies = []
    for s in seeds:
        a = tmaze_fly(s, "odor_a", paired=paired, **kw)
        b = tmaze_fly(s + 50_000, "odor_b", paired=paired, **kw)
        flies.append(dict(seed=s, pi=(a["pi"] + b["pi"]) / 2, halves=[a, b]))
    return dict(paired=paired, flies=flies, pi=[f["pi"] for f in flies])


# --- looming escape ---------------------------------------------------------------------------------------------------
LOOM_SPEEDS = (0.7, 1.4, 3.0, 5.5, 9.0)      # m/s: creeping, crouch-walking, walking, sprinting, a lunge (the 3D game)


def looming_fly(seed: int, speeds=LOOM_SPEEDS, approaches: int = 3, radius: float = 0.28, start: float = 3.0,
                contact: float | None = None, surgery: dict | None = None, params: dict | None = None,
                brain=None) -> dict:
    """An object of `radius` m (default: your body in the 3D game) approaches the fly's head straight on at each speed,
    from `start` m to contact. Returns per speed whether DNp01 crossed the escape threshold before contact, the latency
    from the start of the approach, and how far away the object still was."""
    contact = radius + 0.02 if contact is None else contact
    import kick_the_fly as k

    br = brain or simcore.new_brain(seed=seed, params=params)
    apply_surgery(br, surgery)
    rest(br, 400)
    out = {}
    for v in speeds:
        trials = []
        for _ in range(approaches):
            d, prev_theta, tick, t = start, None, 0, 0.0
            escaped_at, peak = None, 0.0
            while d > contact:
                theta = 2 * math.atan(radius / d)
                if prev_theta is not None:
                    rate = (theta - prev_theta) * 60.0             # rad/s, the game's measure (kick_the_fly._vision)
                    strength = float(np.clip((rate - k.LOOM_MIN) / k.LOOM_FULL, 0, 1))
                    if strength > 0:
                        br.poke("loom", None, strength, recruit=0.6 * strength)
                prev_theta = theta
                for _ in range(_tick_steps(tick)):
                    br._step()
                lvl = br.level("escape")
                peak = max(peak, lvl)
                if escaped_at is None and lvl > k.THRESH["escape"]:
                    escaped_at = t
                    break
                tick += 1
                t += 1 / 60
                d -= v / 60
            trials.append(dict(escaped=escaped_at is not None, latency_s=escaped_at, distance_m=d if escaped_at is not None
                               else None, time_to_contact_s=None if escaped_at is None else (d - contact) / v,
                               peak_level=peak))
            rest(br, 400)
        out[v] = trials
    return dict(seed=seed, radius=radius, start=start, speeds=list(speeds), trials=out)


# --- sugar dose response ----------------------------------------------------------------------------------------------
def offer_sugar(br, dose: float, steps: int = 200, baseline: int = 200) -> dict:
    """Drive `dose` (0..1) of the sugar-pathway GRNs for `steps`; MN9 firing before vs during."""
    g = groups(br)
    mn9 = g["mn9"]
    before = 0
    for _ in range(baseline):
        br._step()
        before += int(np.count_nonzero(br.sim.spikes[mn9]))
    during = 0
    for i in range(steps):
        if dose > 0 and i % 3 == 0:
            br.poke("sweet", None, 0.5, recruit=dose)
        br._step()
        during += int(np.count_nonzero(br.sim.spikes[mn9]))
    b, d = hz(before, len(mn9), baseline), hz(during, len(mn9), steps)
    return dict(dose=dose, mn9_before_hz=b, mn9_hz=d, ratio=d / max(b, 1.0), extended=d >= PER_RATIO * max(b, 1.0))


def sugar_fly(seed: int, doses=(0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0), repeats: int = 2, surgery: dict | None = None,
              params: dict | None = None, brain=None) -> dict:
    br = brain or simcore.new_brain(seed=seed, params=params)
    apply_surgery(br, surgery)
    rest(br, 200)
    out = {d: [] for d in doses}
    rng = np.random.default_rng(seed + 3)
    order = [d for d in doses for _ in range(repeats)]
    rng.shuffle(order)                                  # interleaved doses, like a real dose-response session
    for d in order:
        out[d].append(offer_sugar(br, d))
        rest(br, 200)
    return dict(seed=seed, doses=list(doses), offers=out)


# --- surgery ----------------------------------------------------------------------------------------------------------
def apply_surgery(br, surgery: dict | None) -> None:
    """surgery: {neuron spec: -1 silence | +1 stimulate}, specs as in simcore.rows_of or an assays group name."""
    if not surgery:
        return
    import kick_the_fly as k

    g = groups(br)
    for spec, mode in surgery.items():
        rows = g[spec] if spec in g else simcore.rows_of(br, spec)
        br.set_override(rows, int(mode))
    br.surgery = bool(np.any(br.override))
    assert k.SURGERY_CURRENT
