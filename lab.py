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


def install(menu: ui.Menu) -> None:
    menu.pages["lab"] = page_hub
    menu.pages["lab_params"] = page_params
    menu.pages["lab_assays"] = lambda *a: page_assays(*a)
    menu.pages["lab_validation"] = lambda *a: page_validation(*a)
    ui.TAG_COLORS.setdefault("MODEL", (150, 120, 220))


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


