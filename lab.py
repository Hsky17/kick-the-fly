"""Lab mode screens in the pause menu: the hub, and live parameter controls.

Parameters come in two kinds, tagged on screen:
  MODEL      the LIF simulation's own parameters (connectome/sim.py LIFParams). The wiring is untouched, but the
             validation results were measured at the defaults, so anything changed here is flagged as "modified" in
             the validation dashboard, exports and save states.
  GAME RULE  thresholds the game uses to turn neuron firing into moves, and how looming is converted to drive.
"""
from __future__ import annotations

import time

import pygame

import menu as ui

# name, label, kind (model | rule), default, lo, hi, step, fmt, tooltip
PARAMS = (
    ("noise_std", "Membrane noise", "model", 0.05, 0.0, 0.2, 0.005, "{:.3f}",
     "Random current added to every neuron each 5 ms step. Higher makes the brain flicker more on its own."),
    ("bias", "Tonic drive", "model", 0.20, 0.0, 0.3, 0.005, "{:.3f}",
     "Constant input to every neuron; at 0.20 a neuron rests at 80% of its firing threshold."),
    ("target_rate_hz", "Target rate (Hz)", "model", 5.0, 1.0, 20.0, 0.5, "{:.1f}",
     "The whole-brain firing rate the slow gain controller steers toward."),
    ("ext_gain", "Sensory input gain", "model", 4.0, 1.0, 8.0, 0.1, "{:.1f}",
     "How strongly hits, smells and other stimuli drive the sensory neurons they reach."),
    ("gain_adapt", "Gain adaptation rate", "model", 0.002, 0.0, 0.01, 0.0005, "{:.4f}",
     "How fast the gain controller reacts. 0 freezes synaptic gain where it is."),
    ("thresh.escape", "Dodge: DNp01 above", "rule", 4.0, 1.5, 10.0, 0.1, "{:.1f}x",
     "The giant fiber's firing, as a multiple of its calm rate, that makes the fly dodge."),
    ("thresh.groom", "Groom: aDN1/aDN2 above", "rule", 4.0, 1.5, 10.0, 0.1, "{:.1f}x",
     "Antennal grooming command neurons' firing that logs a GROOM reaction (no body movement)."),
    ("thresh.jump", "Fly away: head-touch DNs above", "rule", 3.0, 1.2, 8.0, 0.1, "{:.1f}x", "Head-touch DN threshold."),
    ("thresh.run", "Run: body-touch DNs above", "rule", 2.4, 1.2, 8.0, 0.1, "{:.1f}x", "Body-touch DN threshold."),
    ("thresh.kick", "Kick: leg-touch DNs above", "rule", 2.0, 1.2, 8.0, 0.1, "{:.1f}x", "Leg-touch DN threshold."),
    ("thresh.walk", "Walk: DNp09 above", "rule", 3.0, 1.2, 8.0, 0.1, "{:.1f}x", "Forward walking threshold."),
    ("thresh.back", "Back up: MDN above", "rule", 3.8, 1.2, 8.0, 0.1, "{:.1f}x", "Moonwalker threshold."),
    ("thresh.turn", "Turn: DNa01/02 R-L above", "rule", 2.1, 0.5, 8.0, 0.1, "{:.1f}", "Steering difference threshold."),
    ("thresh.fly", "Take off: DNg02 above", "rule", 1.58, 1.1, 4.0, 0.02, "{:.2f}x", "Wing-power threshold."),
    ("thresh.fire", "Shoot: DNp35 above", "rule", 3.0, 1.2, 8.0, 0.1, "{:.1f}x", "Duel trigger threshold."),
    ("loom_min", "Looming: ignored below (rad/s)", "rule", 1.5, 0.0, 6.0, 0.1, "{:.1f}",
     "Angular expansion speed below which approaching objects don't drive LPLC2/LC4 at all."),
    ("loom_full", "Looming: full drive span (rad/s)", "rule", 8.0, 1.0, 20.0, 0.5, "{:.1f}",
     "How much faster than the minimum an object must grow to drive the looming detectors fully."),
)
DEFAULTS = {p[0]: p[3] for p in PARAMS}
BY_NAME = {p[0]: p for p in PARAMS}


def apply_to_sim(sim, params: dict) -> None:
    """Model parameters onto one LIFSim (also used for newly spawned flies and headless runs)."""
    for name in ("noise_std", "bias", "target_rate_hz", "ext_gain", "gain_adapt"):
        v = float(params.get(name, DEFAULTS[name]))
        if name == "noise_std":
            old = float(sim.p.noise_std)
            if old > 0:
                sim._noise *= v / old          # the noise bank is pre-scaled at construction
            elif v > 0:
                import numpy as np
                sim._noise = sim.rng.standard_normal(sim._noise.size, dtype=np.float32) * np.float32(v)
        if name == "target_rate_hz":
            sim.target_p = v * sim.p.dt_ms / 1000.0
        setattr(sim.p, name, v)


def apply_rules(params: dict) -> None:
    import kick_the_fly as k2

    for name, value in params.items():
        if name.startswith("thresh."):
            k2.THRESH[name.split(".", 1)[1]] = float(value)
    k2.LOOM_MIN = float(params.get("loom_min", DEFAULTS["loom_min"]))
    k2.LOOM_FULL = float(params.get("loom_full", DEFAULTS["loom_full"]))


def modified(params: dict) -> dict:
    return {k: v for k, v in params.items() if k in DEFAULTS and abs(float(v) - DEFAULTS[k]) > 1e-9}


def page_hub(m: ui.Menu, surf, rect, mouse) -> None:
    host = m.host
    m.text(surf, "LAB", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Research tools. Everything here runs on the same connectome sim as the game.", (rect.x + 24, rect.y + 50),
           ui.LABEL, m.f_small)
    items = [(label, page, tip) for label, page, tip in host.lab_pages() if page in m.pages]
    bw, bh = (rect.w - 72) // 2, 64
    for i, (label, page, tip) in enumerate(items):
        x = rect.x + 24 + (i % 2) * (bw + 24)
        y = rect.y + 90 + (i // 2) * (bh + 16)
        m.button(surf, (x, y, bw, bh), label, (lambda p=page: m.show(p)), id=("lab", page), tip=tip)
    mod = modified(host.lab_params)
    if mod:
        m.text(surf, f"Modified parameters: {', '.join(BY_NAME[k][1] for k in mod)}", (rect.x + 24, rect.bottom - 100),
               ui.AMBER, m.f_small)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("lab", "back"))


def page_params(m: ui.Menu, surf, rect, mouse) -> None:
    host = m.host
    m.text(surf, "PARAMETERS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Applies live to every fly. Validation results were measured at the defaults.", (rect.x + 24, rect.y + 50),
           ui.LABEL, m.f_small)
    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 80 - 70)
    key = "lab_params"
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y + 4 - off
    for name, label, kind, default, lo, hi, step, fmt, tip in PARAMS:
        value = float(host.lab_params.get(name, default))
        row = pygame.Rect(body.x, y, body.w, 40)
        changed = abs(value - default) > 1e-9
        m.text(surf, label, (row.x + 12, row.centery), ui.AMBER if changed else ui.TEXT, m.f_text, "midleft")
        m.chip(surf, (row.x + 330, row.centery - 10), "MODEL" if kind == "model" else "GAME RULE")
        m._register(pygame.Rect(row.x, row.y, 440, row.h), "label", id=("plabel", name),
                    tip=tip + f"  Default {fmt.format(default)}.")
        m.slider(surf, (row.x + 450, row.y + 6, row.w - 470, 28), value, lo, hi, step, fmt,
                 lambda v, n=name: host.set_lab_param(n, v), lambda: None, id=("param", name), tip=tip)
        y += 44
    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None
    m.button(surf, (rect.x + 24, rect.bottom - 58, 200, 42), "Reset to defaults",
             lambda: [host.set_lab_param(n, d) for n, d in DEFAULTS.items()], id=("params", "reset"))
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("params", "back"))


ASSUMPTIONS = (
    ("Raw synapse counts as functional strength proxy",
     "SYNAPSE",
     "The sim treats raw EM synapse counts between neuron pairs as directly proportional to synaptic conductance.",
     "Biological synapses vary widely in vesicle pool size, neurotransmitter release probability, post-synaptic receptor density, and phosphorylation state. Real connection efficacy does not linearly track anatomical contact count.",
     "README.md § Connectome vs Game Rule · connectome/sim.py:LIFParams"),

    ("Uniform synaptic efficacy per connection type",
     "SYNAPSE",
     "All excitatory and inhibitory synapses share fixed base efficacy constants across the whole connectome.",
     "Drosophila synapses exhibit diverse quantal sizes and kinetics across cell types (cholinergic, GABAergic, glutamatergic). Here, sign is assigned from neurotransmitter annotations with uniform base weights.",
     "connectome/sim.py:LIFSim · validation.py"),

    ("Leaky integrate-and-fire (LIF) point neurons",
     "BIOPHYSICS",
     "Each cell body and its entire arbor is condensed into a single isopotential point compartment with tau = 20 ms.",
     "Drosophila neurons have complex non-spiking local computations, passive cable filtering along fine neurites, and compartmentalized local dendritic processing (e.g. in mushroom body lobes and optic lobes) that point-LIF collapses.",
     "connectome/sim.py:LIFParams (tau_m=20ms, dt=5ms)"),

    ("No neurotransmitter or receptor kinetics",
     "DYNAMICS",
     "Synaptic current transfers instantaneously within the discrete 5 ms simulation time-step.",
     "Real ligand-gated and metabotropic receptors have finite activation, desensitization, and clearance timescales (AMPA/nAChR vs slow GABA_B / metabotropic receptors). Slow receptor dynamics are absent.",
     "connectome/sim.py:step()"),

    ("No slow NMDA-like or neuromodulatory states",
     "DYNAMICS",
     "No voltage-dependent ion channel gating, NMDA slow kinetics, or broad volumetric neuromodulator wash.",
     "Neuropeptides and biogenic amines (octopamine, serotonin, dopamine) set global arousal, hunger, and sleep states. Except for modeled reward-driven plasticity, broad state transitions are simplified.",
     "connectome/sim.py · README.md § Limitations"),

    ("Tonic depolarizing bias (0.20 threshold)",
     "TUNING",
     "A constant current bias of 0.20 (80% of threshold) is injected into every neuron to sustain basal activity.",
     "Without background excitation or unmodeled inputs, resting connectome simulations fall completely silent. The tonic bias maintains the biological ~5 Hz spontaneous brain-wide firing rate.",
     "lab.py:PARAMS (bias=0.20) · connectome/sim.py"),

    ("Gaussian membrane noise (sigma = 0.05)",
     "TUNING",
     "Zero-mean Gaussian current noise is added to every neuron at each 5 ms simulation step.",
     "Stochasticity mimics thermal channel noise, spontaneous miniature EPSPs, and unmodeled inputs from sensory organs, preventing artificial deterministic synchronization across identical network paths.",
     "lab.py:PARAMS (noise_std=0.05) · connectome/sim.py"),

    ("Left/Right asymmetry as EM reconstruction artifact risk",
     "DATASET",
     "Asymmetries in synaptic weights or firing between left and right hemibrains reflect both biology and reconstruction noise.",
     "MaleCNS v1.0 EM tracing has variable proofreading depth, staining artifacts, and truncation near slice boundaries. L/R differences may stem from incomplete reconstruction rather than true lateralization.",
     "README.md § Connectome Data · headless.py"),
)


def page_assumptions(m: ui.Menu, surf, rect, mouse) -> None:
    m.text(surf, "MODEL ASSUMPTIONS & SIMPLIFICATIONS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Scientific caveats and approximations distinguishing the simulation from living biology.",
           (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)
    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 80 - 70)
    key = "lab_assumptions"
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y + 4 - off

    category_colors = {
        "SYNAPSE": (100, 180, 240),
        "BIOPHYSICS": (150, 120, 220),
        "DYNAMICS": (240, 160, 60),
        "TUNING": (80, 200, 140),
        "DATASET": (240, 100, 100),
    }

    card_h = 136
    max_w = body.w - 74
    for title, cat, sim_rule, bio_reality, docs in ASSUMPTIONS:
        card = pygame.Rect(body.x, y, body.w - 12, card_h)
        pygame.draw.rect(surf, (24, 28, 38), card, border_radius=8)
        pygame.draw.rect(surf, (45, 52, 68), card, 1, border_radius=8)

        # Category chip
        col = category_colors.get(cat, ui.LABEL)
        chip = pygame.Rect(card.x + 12, card.y + 10, 84, 20)
        pygame.draw.rect(surf, (15, 18, 26), chip, border_radius=4)
        pygame.draw.rect(surf, col, chip, 1, border_radius=4)
        m.text(surf, cat, chip.center, col, m.f_small, "center")

        # Title
        m.text(surf, title, (card.x + 106, card.y + 10), ui.INK, m.f_bold)

        # Sim rule
        m.text(surf, "Sim:", (card.x + 14, card.y + 36), (140, 180, 220), m.f_small)
        m.wrapped(surf, sim_rule, (card.x + 50, card.y + 36), max_w, ui.TEXT, m.f_small, max_lines=2)

        # Biology reality
        m.text(surf, "Bio:", (card.x + 14, card.y + 68), (220, 150, 100), m.f_small)
        m.wrapped(surf, bio_reality, (card.x + 50, card.y + 68), max_w, (185, 190, 200), m.f_small, max_lines=2)

        # Documentation reference
        m.text(surf, "Doc:", (card.x + 14, card.y + 104), ui.LABEL, m.f_small)
        m.text(surf, docs, (card.x + 50, card.y + 104), (130, 160, 210), m.f_small)

        y += card_h + 12

    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None

    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("assumptions", "back"))


def page_asymmetry(m: ui.Menu, surf, rect, mouse) -> None:
    host = m.host
    m.text(surf, "LEFT / RIGHT ASYMMETRY AUDIT", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Audit bilateral differences in connectome structure and spontaneous turning bias.",
           (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)

    body = pygame.Rect(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 80 - 70)
    key = "lab_asymmetry"
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y + 4 - off

    mirror = bool(host.cfg["brain.mirror_weights"]) if hasattr(host, "cfg") else False

    # Status & Context Card
    card_h = 130
    card = pygame.Rect(body.x, y, body.w - 12, card_h)
    pygame.draw.rect(surf, (24, 28, 38), card, border_radius=8)
    pygame.draw.rect(surf, (45, 52, 68), card, 1, border_radius=8)

    chip_col = (240, 160, 60) if mirror else (100, 180, 240)
    chip_txt = "GAME RULE: MIRROR-AVERAGED" if mirror else "CONNECTOME: RAW DATA"
    chip = pygame.Rect(card.x + 12, card.y + 10, 210, 22)
    pygame.draw.rect(surf, (15, 18, 26), chip, border_radius=4)
    pygame.draw.rect(surf, chip_col, chip, 1, border_radius=4)
    m.text(surf, chip_txt, chip.center, chip_col, m.f_small, "center")

    bias_txt = "+1.00 Hz (symmetric weights)" if mirror else "+0.10 Hz (right turn bias)"
    m.text(surf, f"Baseline Turning Bias: {bias_txt}", (card.x + 235, card.y + 12), ui.INK, m.f_bold)

    expl = (
        "In the raw MaleCNS v1.0 connectome, bilateral asymmetries arise from both true biology and uneven electron "
        "microscopy (EM) reconstruction and proofreading depth between hemispheres. Descending steering neurons DNa01 and "
        "DNa02 drive a mild spontaneous rightward turning bias in quiet walking.\n"
        "Mirror-averaging synaptic weights across 77,507 paired bilateral neurons enforces exact structural symmetry. "
        "Because this alters real connectome data, it is tagged strictly as a Game Rule."
    )
    m.wrapped(surf, expl, (card.x + 14, card.y + 40), card.w - 28, ui.TEXT, m.f_small, max_lines=4)
    y += card_h + 14

    # Table Header Card
    hdr_h = 32
    hdr = pygame.Rect(body.x, y, body.w - 12, hdr_h)
    pygame.draw.rect(surf, (32, 38, 52), hdr, border_radius=6)
    m.text(surf, "Key Cell Type", (hdr.x + 14, hdr.centery), ui.INK, m.f_bold, "midleft")
    m.text(surf, "Functional Role", (hdr.x + 120, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "Count L/R", (hdr.x + 370, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "In-Syn L/R", (hdr.x + 470, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "Out-Syn L/R", (hdr.x + 580, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "Calm Rate L/R", (hdr.x + 690, hdr.centery), ui.LABEL, m.f_small, "midleft")
    m.text(surf, "Diff (R-L)", (hdr.right - 14, hdr.centery), ui.LABEL, m.f_small, "midright")
    y += hdr_h + 6

    rows_data = [
        ("DNa01", "Steering descending command", "1 / 1", "393 / 402", "577 / 577", "316 / 311", "532 / 532", "2.4 / 3.0 Hz", "2.2 / 3.2 Hz", "+0.60 Hz", "+1.00 Hz"),
        ("DNa02", "Sharp steering command", "1 / 1", "695 / 716", "1084 / 1084", "339 / 322", "534 / 534", "6.0 / 5.6 Hz", "5.6 / 6.6 Hz", "-0.40 Hz", "+1.00 Hz"),
        ("LC10", "Courtship tracking / fixation", "479 / 481", "78 / 97", "86 / 106", "54 / 60", "72 / 80", "3.8 / 3.9 Hz", "4.0 / 4.1 Hz", "+0.18 Hz", "+0.07 Hz"),
        ("LPLC2", "Rapid looming escape", "94 / 91", "232 / 283", "243 / 295", "93 / 97", "125 / 131", "10.2 / 6.5 Hz", "10.5 / 7.7 Hz", "-3.66 Hz", "-2.79 Hz"),
        ("LC4", "Collision looming avoidance", "112 / 90", "191 / 224", "210 / 244", "79 / 85", "116 / 125", "0.7 / 1.1 Hz", "0.8 / 0.9 Hz", "+0.38 Hz", "+0.13 Hz"),
        ("DNp01", "Braking / backward command", "1 / 1", "558 / 482", "836 / 836", "136 / 125", "200 / 200", "6.6 / 6.2 Hz", "7.6 / 8.6 Hz", "-0.40 Hz", "+1.00 Hz"),
    ]

    row_h = 36
    for t, role, counts, in_raw, in_mir, out_raw, out_mir, r_raw, r_mir, d_raw, d_mir in rows_data:
        rbox = pygame.Rect(body.x, y, body.w - 12, row_h)
        pygame.draw.rect(surf, (20, 24, 33), rbox, border_radius=6)
        m.text(surf, t, (rbox.x + 14, rbox.centery), ui.INK, m.f_bold, "midleft")
        m.text(surf, role, (rbox.x + 120, rbox.centery), (150, 180, 220), m.f_small, "midleft")
        m.text(surf, counts, (rbox.x + 370, rbox.centery), ui.TEXT, m.f_small, "midleft")
        m.text(surf, in_mir if mirror else in_raw, (rbox.x + 470, rbox.centery), ui.TEXT, m.f_small, "midleft")
        m.text(surf, out_mir if mirror else out_raw, (rbox.x + 580, rbox.centery), ui.TEXT, m.f_small, "midleft")
        m.text(surf, r_mir if mirror else r_raw, (rbox.x + 690, rbox.centery), ui.TEXT, m.f_small, "midleft")
        diff_str = d_mir if mirror else d_raw
        diff_val = abs(float(diff_str.replace(" Hz", "")))
        diff_col = ui.GOOD if diff_val < 0.2 else (240, 160, 60)
        m.text(surf, diff_str, (rbox.right - 14, rbox.centery), diff_col, m.f_small, "midright")
        y += row_h + 6

    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None

    btn_txt = "Mirror weights: ON [Rule]" if mirror else "Mirror weights: OFF [Raw]"
    def toggle():
        if hasattr(host, "toggle_mirror_weights"):
            host.toggle_mirror_weights()
    m.button(surf, (rect.x + 24, rect.bottom - 58, 250, 42), btn_txt, toggle,
             id=("asymmetry", "toggle_mirror"), tip="Toggle bilateral weight symmetrization [GAME RULE]")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("asymmetry", "back"))


def install(menu: ui.Menu) -> None:
    menu.pages["lab"] = page_hub
    menu.pages["lab_params"] = page_params
    menu.pages["lab_assays"] = lambda *a: page_assays(*a)
    menu.pages["lab_validation"] = lambda *a: page_validation(*a)
    menu.pages["lab_export"] = lambda *a: page_export(*a)
    menu.pages["lab_protocols"] = lambda *a: page_protocols(*a)
    menu.pages["lab_assumptions"] = page_assumptions
    menu.pages["lab_asymmetry"] = page_asymmetry
    ui.TAG_COLORS.setdefault("MODEL", (150, 120, 220))


def short(path, n: int = 70) -> str:
    """A long path shortened in the middle so it fits on one line."""
    t = str(path)
    return t if len(t) <= n else t[: n // 2 - 2] + "…" + t[-(n // 2 - 1):]


# --- charts ----------------------------------------------------------------------------------------------------------
def draw_chart(m: ui.Menu, surf, rect: pygame.Rect, xs, series, x_label: str, y_label: str, y_max: float | None = None,
               x_fmt="{:g}", connect: bool = True) -> None:
    """Line chart with 95% CI bands. series: [(label, color, [mean_ci dicts per x])]."""
    import math

    pygame.draw.rect(surf, (12, 14, 20), rect, border_radius=8)
    plot = pygame.Rect(rect.x + 52, rect.y + 44, rect.w - 70, rect.h - 82)
    vals = [c["hi"] if not math.isnan(c.get("hi", float("nan"))) else c["mean"] for _, _, cs in series for c in cs
            if not math.isnan(c["mean"])]
    top = y_max if y_max is not None else max(vals + [1e-6]) * 1.1
    for k in range(5):
        y = plot.bottom - plot.h * k / 4
        pygame.draw.line(surf, (36, 40, 50), (plot.x, y), (plot.right, y))
        m.text(surf, f"{top * k / 4:.2g}", (plot.x - 6, y), ui.LABEL, m.f_small, "midright")
    n = len(xs)

    def px(i):
        return plot.x + (plot.w * (i + 0.5) / n)

    def py(v):
        return plot.bottom - plot.h * min(max(v / top, 0), 1)

    for i, x in enumerate(xs):
        m.text(surf, x_fmt.format(x) if isinstance(x, (int, float)) else str(x), (px(i), plot.bottom + 6), ui.LABEL,
               m.f_small, "midtop")
    for li, (label, col, cs) in enumerate(series):
        pts = []
        for i, c in enumerate(cs):
            if math.isnan(c["mean"]):
                continue
            if not math.isnan(c.get("lo", float("nan"))):
                pygame.draw.line(surf, col, (px(i), py(c["lo"])), (px(i), py(c["hi"])), 2)
            pts.append((px(i), py(c["mean"])))
            pygame.draw.circle(surf, col, (int(px(i)), int(py(c["mean"]))), 4)
        if len(pts) > 1 and connect:
            pygame.draw.lines(surf, col, False, pts, 2)
        m.text(surf, label, (rect.right - 12 - li * 190, rect.y + 6), col, m.f_small, "topright")
    m.text(surf, x_label, (plot.centerx, rect.bottom - 16), ui.TEXT, m.f_small, "midtop")
    m.text(surf, y_label, (rect.x + 6, rect.y + 2), ui.TEXT, m.f_small)


def _ci(c: dict, fmt="{:.2f}") -> str:
    import math

    if c["n"] == 0 or math.isnan(c["mean"]):
        return "n/a"
    if math.isnan(c["lo"]):
        return fmt.format(c["mean"])
    return f"{fmt.format(c['mean'])} ± {fmt.format((c['hi'] - c['lo']) / 2)}"


# --- assays and repeated trials ---------------------------------------------------------------------------------------
def surgery_options() -> list[tuple[str, dict | None]]:
    import kick_the_fly as k

    out = [("none", None)]
    for label, (kind, names) in k.SURGERY:
        if kind == "type":
            spec = {"type:" + ",".join(names): -1}
        elif kind == "group":
            spec = {n: -1 for n in names}
        elif kind == "prefix":
            spec = {"prefix:" + ",".join(names): -1}
        elif kind == "pop":
            spec = {n: -1 for n in names}
        else:
            continue
        out.append((label, spec))
    return out


class LabState:
    def __init__(self):
        self.kind = "tmaze"
        self.flies = 6
        self.surgery_i = 0
        self.stimulate = False
        self.job = None
        self.result = None
        self.validation_job = None


def _state(m) -> LabState:
    if not hasattr(m, "lab_state"):
        m.lab_state = LabState()
    return m.lab_state


def page_assays(m: ui.Menu, surf, rect, mouse) -> None:
    import labjobs
    import labstats

    st, host = _state(m), m.host
    m.text(surf, "ASSAYS AND REPEATED TRIALS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Each fly is a fresh, untrained brain with its own seed. Your saved training memory isn't used or "
                 "changed.", (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)
    x0, y = rect.x + 24, rect.y + 78
    busy = st.job is not None and st.job.running
    m.text(surf, "Assay", (x0, y + 15), ui.TEXT, m.f_text, "midleft")
    m.segmented(surf, (x0 + 110, y, 520, 30), [labjobs.ASSAY_LABEL[k] for k in labjobs.ASSAYS],
                labjobs.ASSAYS.index(st.kind), lambda i: setattr(st, "kind", labjobs.ASSAYS[i]), id="assay_kind",
                enabled=not busy)
    y += 40
    m.text(surf, "Flies", (x0, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (x0 + 110, y, 300, 30), st.flies, 2, 30, 1, "{:.0f}", lambda v: setattr(st, "flies", int(v)),
             lambda: None, id="assay_flies", enabled=not busy,
             tip="How many flies (seeds) to run. More gives tighter confidence intervals and takes longer.")
    base = int(host.cfg["brain.seed"])
    m.text(surf, f"seeds {base + 2000}-{base + 2000 + st.flies - 1}", (x0 + 430, y + 15), ui.LABEL, m.f_small, "midleft")
    y += 40
    opts = surgery_options()
    label, spec = opts[st.surgery_i]
    m.text(surf, "Surgery", (x0, y + 15), ui.TEXT, m.f_text, "midleft")
    m.button(surf, (x0 + 110, y, 44, 30), "<", lambda: setattr(st, "surgery_i", (st.surgery_i - 1) % len(opts)),
             id="surg<", enabled=not busy)
    m.text(surf, label, (x0 + 170, y + 15), ui.INK if spec else ui.LABEL, m.f_bold, "midleft")
    m.button(surf, (x0 + 520, y, 44, 30), ">", lambda: setattr(st, "surgery_i", (st.surgery_i + 1) % len(opts)),
             id="surg>", enabled=not busy)
    if spec:
        m.segmented(surf, (x0 + 580, y, 220, 30), ["Silence", "Stimulate"], int(st.stimulate),
                    lambda i: setattr(st, "stimulate", bool(i)), id="surg_mode", enabled=not busy)
    y += 40
    if spec:
        m.text(surf, "Each fly also runs unperturbed with the same seed as its control; the results show both and a "
                     "paired test.", (x0, y), ui.LABEL, m.f_small)
    y += 24

    def start():
        surgery = {k_: (1 if st.stimulate else -1) for k_ in spec} if spec else None
        seeds = [base + 2000 + i for i in range(st.flies)]
        params = modified(host.lab_params) and dict(host.lab_params) or None
        st.result = None
        st.job = labjobs.Job(st.kind, seeds, surgery=surgery, params=params).start()
        st.job.surgery_label = label

    if busy:
        j = st.job
        frac = j.done / max(1, j.total)
        pygame.draw.rect(surf, (30, 36, 48), (x0, y, rect.w - 220, 14), border_radius=7)
        pygame.draw.rect(surf, ui.AMBER, (x0, y, max(8, int((rect.w - 220) * frac)), 14), border_radius=7)
        m.text(surf, f"{j.done}/{j.total} flies  ·  {time.time() - j.started:.0f}s  ·  {j.workers} worker processes",
               (x0, y + 20), ui.TEXT, m.f_small)
        m.button(surf, (rect.right - 170, y - 8, 146, 36), "Cancel", lambda: setattr(j, "cancelled", True),
                 id="assay_cancel", style="danger")
    else:
        if st.job is not None and st.job.result is not None and st.result is not st.job.result:
            st.result = st.job.result
            host.last_lab_result = st.result
        m.button(surf, (x0, y - 6, 180, 40), "Run", start, id="assay_run", style="primary",
                 tip="Runs in background worker processes while the game stays paused.")
        if st.job is not None and st.job.error:
            m.text(surf, st.job.error, (x0 + 200, y + 14), ui.BAD, m.f_small, "midleft")
        if st.result is not None:
            m.button(surf, (x0 + 200, y - 6, 200, 40), "Export results", lambda: host.export_lab_result(st.result),
                     id="assay_export", tip="Save these results as JSON and CSV in the exports folder.")
    y += 40
    if st.result is not None and not busy:
        draw_assay_result(m, surf, pygame.Rect(rect.x + 24, y, rect.w - 48, rect.bottom - 70 - y), st.result)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("assays", "back"))


def draw_assay_result(m: ui.Menu, surf, area: pygame.Rect, res: dict) -> None:
    import labstats

    kind, t, c = res["kind"], res["treated"], res.get("control")
    surg = res.get("surgery")
    tl = "with surgery" if surg else "flies"
    chart = pygame.Rect(area.x, area.y, area.w // 2 - 10, area.h)
    tx = area.x + area.w // 2 + 10
    y = area.y
    m.text(surf, f"{res['label']}  ·  n = {len(res['seeds'])} flies", (tx, y), ui.INK, m.f_bold)
    y += 26
    if kind == "tmaze":
        xs, pts = (["with surgery", "control"], [t["pi"], c["pi"]]) if c else (["all flies"], [t["pi"]])
        draw_chart(m, surf, chart, xs, [("mean and 95% CI", ui.ACCENT, pts)], "", "performance index", y_max=1.0,
                   connect=False)
        rows = [("performance index", _ci(t["pi"]), _ci(c["pi"]) if c else ""),
                ("fear of CS+", _ci(t["fear_cs_plus"]), _ci(c["fear_cs_plus"]) if c else ""),
                ("fear of CS-", _ci(t["fear_cs_minus"]), _ci(c["fear_cs_minus"]) if c else ""),
                ("approach MBONs to CS+ (Hz)", _ci(t["mbon_cs_plus_hz"], "{:.1f}"), _ci(c["mbon_cs_plus_hz"], "{:.1f}") if c else ""),
                ("approach MBONs to CS- (Hz)", _ci(t["mbon_cs_minus_hz"], "{:.1f}"), _ci(c["mbon_cs_minus_hz"], "{:.1f}") if c else "")]
    elif kind == "looming":
        xs = [r["speed"] for r in t["rows"]]
        series = [(tl, ui.ACCENT, [r["escape_probability"] for r in t["rows"]])]
        if c:
            series.append(("control", (200, 200, 200), [r["escape_probability"] for r in c["rows"]]))
        draw_chart(m, surf, chart, xs, series, "approach speed (m/s)", "escape probability", y_max=1.0)
        rows = [(f"{r['speed']:g} m/s", f"{r['escapes']}/{r['approaches']} escaped, latency {_ci(r['latency_s'])} s",
                 (f"{cr['escapes']}/{cr['approaches']}" if c else "")) for r, cr in zip(t["rows"], c["rows"] if c else t["rows"])]
    else:
        xs = [r["dose"] for r in t["rows"]]
        series = [(tl, ui.ACCENT, [r["mn9_ratio"] for r in t["rows"]])]
        if c:
            series.append(("control", (200, 200, 200), [r["mn9_ratio"] for r in c["rows"]]))
        draw_chart(m, surf, chart, xs, series, "sugar dose (share of sugar-pathway GRNs)", "MN9 firing x before",
                   x_fmt="{:.0%}")
        rows = [(f"{r['dose']:.0%}", f"MN9 x{_ci(r['mn9_ratio'])}, extended {r['extensions']}/{r['offers']}",
                 (f"x{cr['mn9_ratio']['mean']:.2f}" if c else "")) for r, cr in zip(t["rows"], c["rows"] if c else t["rows"])]
    col_a = tx + 190
    col_b = tx + (area.w // 2 - 10) - 120
    if c:
        m.text(surf, "surgery", (col_a, y), ui.LABEL, m.f_small)
        m.text(surf, "control", (col_b, y), ui.LABEL, m.f_small)
        y += 18
    else:
        m.text(surf, "mean ± 95% CI half-width", (col_a, y), ui.LABEL, m.f_small)
        y += 18
    for label, a, b in rows:
        m.text(surf, label, (tx, y), ui.TEXT, m.f_small)
        if c:
            m.wrapped(surf, a, (col_a, y), col_b - col_a - 10, ui.INK, m.f_small, 1)
            m.text(surf, b, (col_b, y), ui.INK, m.f_small)
        else:
            m.wrapped(surf, a, (col_a, y), area.right - col_a, ui.INK, m.f_small, 1)
        y += 20
    if c:
        cmp_ = res["comparison"]["overall"]
        y += 8
        m.text(surf, f"Surgery vs same-seed control: mean difference {cmp_['mean_difference']:+.3f} (n={cmp_['n']} pairs)",
               (tx, y), ui.INK, m.f_small)
        m.text(surf, f"{cmp_['test']}: {labstats.fmt_p(cmp_['p_value'])}   (paired t: {labstats.fmt_p(cmp_.get('t_p_value'))})",
               (tx, y + 18), ui.AMBER if cmp_["p_value"] < 0.05 else ui.TEXT, m.f_small)
        m.text(surf, f"surgery: {', '.join(f'{k} {v:+d}' for k, v in res['surgery'].items())}", (tx, y + 36), ui.LABEL, m.f_small)


# --- validation dashboard ---------------------------------------------------------------------------------------------
def page_validation(m: ui.Menu, surf, rect, mouse) -> None:
    import labstats
    import validation

    st, host = _state(m), m.host
    m.text(surf, "VALIDATION", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    job = st.validation_job
    if job is not None and not job["thread"].is_alive() and job.get("result"):
        host.refresh_validation()
        st.validation_job = job = None
    res, source = validation.load_results()
    if res is None:
        m.text(surf, "No results yet. Run the suite (a few minutes).", (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)
    else:
        m.text(surf, f"Kick the Fly {res['app_version']}, {res['created']}, {source}  ·  seeds {res['seeds'][0]}-"
                     f"{res['seeds'][-1]} (n={len(res['seeds'])})  ·  thresholds chosen for this release, not from the papers",
               (rect.x + 24, rect.y + 50), ui.LABEL, m.f_small)
    if modified(host.lab_params):
        m.text(surf, "Parameters are modified: these results were measured at the defaults.", (rect.right - 24, rect.y + 20),
               ui.AMBER, m.f_small, "topright")
    body = pygame.Rect(rect.x + 16, rect.y + 76, rect.w - 32, rect.h - 76 - 70)
    off = int(m.scroll.get("lab_validation", 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y - off
    for t in (res or {}).get("tests", []):
        card = pygame.Rect(body.x, y, body.w - 12, 156)
        pygame.draw.rect(surf, (26, 30, 40), card, border_radius=10)
        ok = t["passed"]
        chip = pygame.Rect(card.x + 12, card.y + 12, 64, 24)
        pygame.draw.rect(surf, ui.GOOD if ok else ui.BAD, chip, border_radius=6)
        m.text(surf, "PASS" if ok else "FAIL", chip.center, (10, 12, 16), m.f_bold, "center")
        m.text(surf, t["name"], (card.x + 90, card.y + 12), ui.INK, m.f_bold)
        m.text(surf, t["claim"][:140], (card.x + 90, card.y + 36), ui.TEXT, m.f_small)
        mm = t["measured"]
        if "drive_ratio_mean" in mm:
            meas = (f"{t['readout_label']}: x{mm['drive_ratio_mean']:.2f} ± {mm['drive_ratio_sd']:.2f} driving "
                    f"{t['drive_label']}  vs  x{mm['control_ratio_mean']:.2f} ± {mm['control_ratio_sd']:.2f} for "
                    f"{t['control_label']}  ·  {labstats.fmt_p(mm['p_value'])}")
        else:
            meas = (f"PI {mm['pi_mean']:.2f} ± {mm['pi_sd']:.2f} vs unpaired {mm['control_pi_mean']:.2f} ± "
                    f"{mm['control_pi_sd']:.2f}  ·  fear CS+ {mm['fear_cs_plus']:.2f} vs CS- {mm['fear_cs_minus']:.2f}  ·  "
                    f"approach MBONs {mm['approach_mbon_cs_plus_hz']:.1f} vs {mm['approach_mbon_cs_minus_hz']:.1f} Hz  ·  "
                    f"{labstats.fmt_p(mm['p_value'])}")
        yy = m.wrapped(surf, meas, (card.x + 90, card.y + 58), card.w - 110, ui.INK, m.f_small, 2)
        m.text(surf, f"Pass if: {t['criteria']}", (card.x + 90, yy + 4), ui.LABEL, m.f_small)
        m.text(surf, t["citation"], (card.x + 90, yy + 24), (150, 180, 220), m.f_small)
        if t.get("note"):
            m._register(pygame.Rect(card.x, card.y, card.w, card.h), "label", id=("vnote", t["id"]), tip=t["note"])
            m.text(surf, "hover for notes", (card.right - 12, card.y + 12), ui.DIM, m.f_small, "topright")
        y += 164
    m.content_h["lab_validation"] = max(0, y + off - body.bottom)
    surf.set_clip(prev)
    m.clip = None
    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, rect.bottom - 44, rect.w - 420, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, rect.bottom - 44, max(8, int((rect.w - 420) * frac)), 12), border_radius=6)
        m.text(surf, f"running: {job['done']}/{job['total']}" + (f"  error: {job['error']}" if job.get("error") else ""),
               (rect.x + 24, rect.bottom - 28), ui.TEXT, m.f_small)
    else:
        m.button(surf, (rect.x + 24, rect.bottom - 58, 240, 42), "Run validation now", lambda: start_validation(m),
                 id="val_run", tip="Runs every test on this PC with the default parameters (a few minutes). The result "
                                   "replaces the build's bundled one on this PC.")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("val", "back"))


def start_validation(m: ui.Menu) -> None:
    import threading

    import labjobs
    import validation

    st = _state(m)
    job = dict(done=0, total=1, result=None, error=None)

    def work():
        try:
            res = validation.run(workers=labjobs.default_workers(),
                                 progress=lambda d, n, label: job.update(done=d, total=n))
            validation.save_results(res)
            job["result"] = res
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="validation", daemon=True)
    job["thread"].start()
    st.validation_job = job




# --- record and export ------------------------------------------------------------------------------------------------
RECORD_GROUPS = (
    ("Giant fiber DNp01", "dnp01"), ("Looming detectors LPLC2, LC4", "loom"), ("Touch neurons", "touch"),
    ("Reaction descending neurons", "reaction_dns"), ("All descending neurons", "superclass:descending_neuron"),
    ("Proboscis motor neuron MN9", "mn9"), ("Grooming command aDN1/aDN2", "adn"), ("Antennal JO-C/E", "jo_ce"),
    ("Sugar-pathway taste neurons", "sweet"), ("Kenyon cells", "prefix:KC"), ("Mushroom body output neurons", "prefix:MBON"),
    ("Reward dopamine PAM", "reward"), ("Punishment dopamine PPL1", "punish"), ("Leg motor neurons", "mn_legs"),
    ("Whole brain (large files)", "whole brain"),
)


def resolve_group(br, spec: str):
    import numpy as np

    import assays
    import kick_the_fly as k
    import simcore

    g = assays.groups(br)
    if spec in g:
        return g[spec]
    if spec == "touch":
        return np.concatenate([simcore.rows_of(br, n) for n in k.TOUCH])
    if spec == "reaction_dns":
        return np.concatenate([simcore.rows_of(br, n) for n, *_ in k.MOTOR])
    return simcore.rows_of(br, spec)


def page_export(m: ui.Menu, surf, rect, mouse) -> None:
    import recorder

    st, host = _state(m), m.host
    if not hasattr(st, "rec_pick"):
        st.rec_pick, st.rec_seconds = {"dnp01", "loom", "reaction_dns"}, 10
    m.text(surf, "RECORD AND EXPORT", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Records spike times from the fly your brain panel shows, live, while you play. Files: spikes and rates "
                 "as CSV and npz, plus a metadata JSON.", (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)
    y = rect.y + 84
    options = list(RECORD_GROUPS)
    insp = getattr(host, "inspect", None)
    if insp:
        options.append((f"Inspected type {insp['type']}", "type:" + insp["type"]))
    col_w = (rect.w - 48) // 2
    for i, (label, spec) in enumerate(options):
        x = rect.x + 24 + (i % 2) * col_w
        yy = y + (i // 2) * 38
        on = spec in st.rec_pick
        m.toggle(surf, (x, yy, 90, 30), on, lambda v, s=spec: (st.rec_pick.add(s) if v else st.rec_pick.discard(s)),
                 id=("rec", spec))
        m.text(surf, label, (x + 100, yy + 15), ui.TEXT, m.f_text, "midleft")
    y += ((len(options) + 1) // 2) * 38 + 12
    m.text(surf, "Duration", (rect.x + 24, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (rect.x + 140, y, 360, 30), st.rec_seconds, 1, 60, 1, "{:.0f} s",
             lambda v: setattr(st, "rec_seconds", int(v)), lambda: None, id="rec_seconds")
    y += 50
    rec = getattr(host, "recording", None)
    if rec is None:
        m.button(surf, (rect.x + 24, y, 260, 44), "Start recording and resume", lambda: host.start_recording(
            [(lbl, s) for lbl, s in options if s in st.rec_pick], st.rec_seconds), style="primary", id="rec_start",
            enabled=bool(st.rec_pick), tip="Closes the menu; the recording stops by itself after the duration.")
    else:
        m.button(surf, (rect.x + 24, y, 200, 44), "Stop and save", host.stop_recording, style="danger", id="rec_stop")
    m.text(surf, f"Saved to {short(recorder.exports_dir(), 110)}", (rect.x + 24, y + 56), ui.LABEL, m.f_small)
    last = getattr(host, "last_export", None)
    if last:
        m.text(surf, f"Last: {short(last, 110)}", (rect.x + 24, y + 76), ui.GOOD, m.f_small)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("export", "back"))


# --- protocols --------------------------------------------------------------------------------------------------------
def protocol_files() -> list:
    import sys
    from pathlib import Path

    import paths

    user = paths.get().data_dir / "protocols"
    roots = [user, Path(getattr(sys, "_MEIPASS", "")) / "protocols", Path(__file__).resolve().parent / "protocols"]
    seen, out = set(), []
    for root in roots:
        if str(root) and root.is_dir():
            for f in sorted(root.glob("*.y*ml")):
                if f.name not in seen:
                    seen.add(f.name)
                    out.append(f)
    return out


def page_protocols(m: ui.Menu, surf, rect, mouse) -> None:
    import threading

    import labjobs
    import paths
    import protocol

    st = _state(m)
    m.text(surf, "PROTOCOLS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, f"YAML experiment files. Put your own in {short(paths.get().data_dir / 'protocols', 60)}. Headless: "
                 "KickTheFly --headless --protocol FILE", (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)
    job = getattr(st, "proto_job", None)
    busy = job is not None and job["thread"].is_alive()
    y = rect.y + 84
    for f in protocol_files()[:12]:
        try:
            p = protocol.load(f)
            desc = (f"assay {p['assay']}, " if "assay" in p else f"{len(p['stimuli'])} stimuli, {len(p['recordings'])} "
                    f"recordings, ") + f"{len(p['seeds'])} fly(s)" + (", with surgery + control" if p.get("surgery") else "")
            err = None
        except Exception as e:
            desc, err, p = str(e), True, None
        row = pygame.Rect(rect.x + 24, y, rect.w - 48, 42)
        pygame.draw.rect(surf, (26, 30, 40), row, border_radius=8)
        m.text(surf, f.name, (row.x + 12, row.y + 4), ui.INK, m.f_bold)
        m.text(surf, desc[:120], (row.x + 12, row.y + 23), ui.BAD if err else ui.LABEL, m.f_small)

        def start(p=p):
            j = dict(done=0, total=1, folder=None, error=None, name=p["name"])

            def work():
                try:
                    j["folder"] = protocol.run(p, workers=labjobs.default_workers(),
                                               progress=lambda d, n: j.update(done=d, total=n))
                except Exception as e:
                    j["error"] = f"{type(e).__name__}: {e}"

            j["thread"] = threading.Thread(target=work, name="protocol", daemon=True)
            j["thread"].start()
            st.proto_job = j

        m.button(surf, (row.right - 110, row.y + 6, 96, 30), "Run", start, id=("proto", f.name),
                 enabled=not err and not busy, font=m.f_small)
        y += 50
    if job is not None:
        if busy:
            frac = job["done"] / max(1, job["total"])
            pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, rect.bottom - 100, rect.w - 220, 12), border_radius=6)
            pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, rect.bottom - 100, max(8, int((rect.w - 220) * frac)), 12),
                             border_radius=6)
            m.text(surf, f"running {job['name']}: {job['done']}/{job['total']}", (rect.x + 24, rect.bottom - 84), ui.TEXT, m.f_small)
        elif job["error"]:
            m.text(surf, job["error"], (rect.x + 24, rect.bottom - 90), ui.BAD, m.f_small)
        elif job["folder"]:
            m.text(surf, f"Done: {short(job['folder'], 110)}", (rect.x + 24, rect.bottom - 90), ui.GOOD, m.f_small)
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("proto", "back"))
