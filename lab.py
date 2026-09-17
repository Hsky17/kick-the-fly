"""Lab mode screens in the pause menu: the hub, and live parameter controls.

Parameters come in two kinds, tagged on screen:
  MODEL      the LIF simulation's own parameters (connectome/sim.py LIFParams). The wiring is untouched, but the
             validation results were measured at the defaults, so anything changed here is flagged as "modified" in
             the validation dashboard, exports and save states.
  GAME RULE  thresholds the game uses to turn neuron firing into moves, and how looming is converted to drive.
"""
from __future__ import annotations

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


def install(menu: ui.Menu) -> None:
    menu.pages["lab"] = page_hub
    menu.pages["lab_params"] = page_params
    ui.TAG_COLORS.setdefault("MODEL", (150, 120, 220))
