"""Lab screens that change the connectome itself rather than driving neurons.

  Synapse threshold    drop every connection reconstructed with fewer than N synapses, live, and report the
                       threshold at which each validated behavior breaks
  Sign flips           flip the sign of neurons whose neurotransmitter prediction the dataset is least sure of
  Inhibition block     scale every predicted-inhibitory synapse, 0-100%

The numbers each one acts on (synapse counts, neurotransmitter predictions and their confidence) are the dataset's.
The sliders, the trial counts and how a severity maps onto them are this game's choices, and both are said so on
screen.
"""
from __future__ import annotations

import threading

import pygame

from kickthefly.ui import menu as ui

# (key, label, draw function). Each feature adds its own tab here.
TABS: list[tuple[str, str, object]] = []


def tab(key: str, label: str):
    def wrap(fn):
        TABS.append((key, label, fn))
        return fn
    return wrap


def _st(m):
    from kickthefly.lab import lab

    st = lab._state(m)
    if not hasattr(st, "wiring_tab"):
        st.wiring_tab = "threshold"
        st.min_syn = 1
        st.sweep_job = None
        st.sweep_result = None
        st.flip_conf = 0.5
        st.flip_job = None
        st.flip_result = None
        st.flip_trials = 10
        st.flip_seeds = 10
        st.inhibition = 100
        st.inhibition_before = None
    return st


def page(m: ui.Menu, surf, rect, mouse) -> None:
    st, host = _st(m), m.host
    m.text(surf, "CONNECTOME ROBUSTNESS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Change the wiring itself and re-run the behaviors. Everything here acts on the dataset's own "
                 "synapse counts and neurotransmitter predictions.", (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)
    x = rect.x + 24
    for key, label, _ in TABS:
        r = pygame.Rect(x, rect.y + 72, 200, 34)
        m.button(surf, r, label, (lambda k=key: setattr(st, "wiring_tab", k)), id=("wtab", key),
                 active=st.wiring_tab == key, style="normal")
        x += 210
    body = pygame.Rect(rect.x + 16, rect.y + 116, rect.w - 32, rect.h - 116 - 70)
    busy = getattr(host, "wiring_busy", "")
    if busy:
        m.text(surf, f"applying to every fly: {busy} …", (rect.right - 24, rect.y + 84), ui.AMBER, m.f_small,
               "topright")
    draw = next((fn for key, _, fn in TABS if key == st.wiring_tab), TABS[0][2] if TABS else None)
    if draw is not None:
        draw(m, surf, body, st, host)
    w = getattr(host, "wiring", None)
    if w is not None and not w.is_identity:
        m.text(surf, f"live: {w.label()}", (rect.x + 24, rect.bottom - 48), ui.AMBER, m.f_small)
        m.button(surf, (rect.x + 24, rect.bottom - 34, 200, 28), "Restore the connectome",
                 lambda: _restore(host), id="wire_reset", style="danger")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("wire", "back"))


def _restore(host) -> None:
    from kickthefly.sim.wiring import Wiring

    host.set_wiring(Wiring())


# --- synapse threshold ------------------------------------------------------------------------------------------
@tab("threshold", "Synapse threshold")
def _tab_threshold(m, surf, body, st, host) -> None:
    from kickthefly.lab import robustness
    from kickthefly.sim import wiring as wiring_mod
    from kickthefly.sim.wiring import Wiring

    y = body.y
    m.text(surf, "Drop every connection the reconstruction found fewer than N synapses for, then watch the fly and "
                 "re-run the behaviors.", (body.x + 8, y), ui.TEXT, m.f_small)
    m.text(surf, f"Weak contacts are the ones most likely to be reconstruction noise. The brain pack already drops "
                 f"anything below {robustness.MIN_PACK_SYNAPSES} synapses, so 1-{robustness.MIN_PACK_SYNAPSES} "
                 f"change nothing.", (body.x + 8, y + 18), ui.LABEL, m.f_small)
    m.chip(surf, (body.x + 8, y + 40), "CONNECTOME")
    m.text(surf, "the synapse counts are the dataset's; which thresholds to try is a game choice",
           (body.x + 120, y + 48), ui.LABEL, m.f_small, "midleft")
    y += 72
    m.text(surf, "Minimum synapses", (body.x + 8, y + 15), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (body.x + 200, y, body.w - 420, 30), st.min_syn, 1, 15, 1, "{:.0f}",
             lambda v: setattr(st, "min_syn", int(v)), lambda: None, id="min_syn",
             tip="Connections with fewer synapses than this are set to zero weight. The neurons stay; only the weak "
                 "wiring between them goes.")
    y += 44
    try:
        stats = wiring_mod.threshold_stats(int(st.min_syn))
    except FileNotFoundError:
        m.text(surf, "needs the brain pack", (body.x + 8, y), ui.BAD, m.f_small)
        return
    m.text(surf, f"At >= {stats['threshold']} synapses: {stats['connections_dropped']:,} of "
                 f"{stats['connections']:,} connections dropped ({stats['connections_dropped_share']:.1%}), "
                 f"{stats['synapses_dropped']:,} of {stats['synapses']:,} synapses",
           (body.x + 8, y), ui.INK, m.f_text)
    m.text(surf, f"{stats['neurons_touched']:,} of {stats['neurons']:,} neurons lose at least one connection; "
                 f"{stats['neurons_cut_off']:,} lose every input, {stats['neurons_silenced_output']:,} lose every "
                 f"output", (body.x + 8, y + 24), ui.TEXT, m.f_small)
    y += 56
    live = getattr(host, "wiring", None)
    applied = live is not None and live.min_synapses == int(st.min_syn)
    m.button(surf, (body.x + 8, y, 260, 40), "Apply to every fly" if not applied else "Applied",
             lambda: host.set_wiring(Wiring(min_synapses=int(st.min_syn), flip_rows=live.flip_rows if live else (),
                                            inhibition_scale=live.inhibition_scale if live else 1.0)),
             id="apply_thresh", style="primary", enabled=not applied and not getattr(host, "wiring_busy", ""),
             tip="Rebuilds every live fly's synapse matrix. Takes about a second per fly.")
    job, res = st.sweep_job, st.sweep_result
    if job is not None and not job["thread"].is_alive():
        st.sweep_result = res = job.get("result") or res
        if job.get("error"):
            m.text(surf, f"error: {job['error']}", (body.x + 290, y + 20), ui.BAD, m.f_small, "midleft")
        st.sweep_job = job = None
    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (body.x + 290, y + 14, body.w - 320, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (body.x + 290, y + 14, max(8, int((body.w - 320) * frac)), 12),
                         border_radius=6)
        m.text(surf, job["label"], (body.x + 290, y + 30), ui.LABEL, m.f_small)
    else:
        m.button(surf, (body.x + 290, y, 300, 40), "Run the threshold report", lambda: _start_sweep(m, st),
                 id="run_sweep", tip="Re-runs each validated behavior at thresholds 1, 4, 5, 6, 8 and 10 on the "
                                     "validation seeds, with the validation suite's own pass criteria, and reports "
                                     "the threshold at which each one breaks. Takes several minutes.")
    y += 52
    if res:
        _draw_sweep(m, surf, pygame.Rect(body.x + 8, y, body.w - 16, body.bottom - y), st, res)


def _start_sweep(m, st) -> None:
    from kickthefly.lab import labjobs, robustness

    job = dict(done=0, total=1, label="starting", result=None, error=None)

    def work():
        try:
            job["result"] = robustness.threshold_sweep(
                workers=labjobs.default_workers(),
                progress=lambda d, n, label: job.update(done=d, total=n, label=label))
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="threshold-sweep", daemon=True)
    job["thread"].start()
    st.sweep_job = job


def _draw_sweep(m, surf, area, st, res) -> None:
    m.text(surf, f"Threshold report  ·  seeds {res['seeds'][0]}-{res['seeds'][-1]}  ·  {res['seconds']:.0f}s  ·  "
                 f"validation's own pass criteria", (area.x, area.y), ui.INK, m.f_small)
    y = area.y + 22
    head = ["behavior"] + [f">={t}" for t in res["thresholds"]] + ["breaks at"]
    xs = [area.x] + [area.x + 300 + i * 56 for i in range(len(res["thresholds"]))] + [area.right - 120]
    for hx, label in zip(xs, head):
        m.text(surf, label, (hx, y), ui.LABEL, m.f_small)
    y += 20
    for test_id, b in res["breaks"].items():
        m.text(surf, b["name"][:44], (area.x, y), ui.TEXT, m.f_small)
        for i, t in enumerate(res["thresholds"]):
            step = next((s for s in res["steps"] if s["threshold"] == t), None)
            ok = step and step["behaviors"].get(test_id, {}).get("passed")
            m.text(surf, "pass" if ok else "fail", (xs[i + 1], y), ui.GOOD if ok else ui.BAD, m.f_small)
        m.text(surf, f">= {b['breaks_at']}" if b["breaks_at"] else "holds", (xs[-1], y),
               ui.BAD if b["breaks_at"] else ui.GOOD, m.f_small)
        y += 20
    y += 6
    m.text(surf, "A behavior that breaks at a low threshold depends on connections the reconstruction saw only a few "
                 "synapses of.", (area.x, y), ui.LABEL, m.f_small)
    m.button(surf, (area.x, min(y + 20, area.bottom - 34), 200, 30), "Export CSV + JSON",
             lambda: _export(m, res), id="sweep_export")


def _export(m, res) -> None:
    import time

    from kickthefly.lab import recorder, robustness

    folder = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{res['kind']}"
    robustness.save(res, folder)
    m.host.last_export = str(folder)
    m.flash(f"saved to {folder.name} in exports", ui.GOOD)

