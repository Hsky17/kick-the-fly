"""Lab configuration screen for Targeted Optogenetics Laser.

Target cell types directly in-world with activating or silencing beams.
Scientific note: This models targeted cellular stimulation and silencing.
It does NOT claim Gal4/UAS driver line or opsin-specific kinetics.
"""
from __future__ import annotations

import pygame

from kickthefly.ui import menu as ui
from kickthefly.lab.laser import DEFAULT_TARGETS, LaserState


def _get_laser(host) -> LaserState:
    if not hasattr(host, "laser_state"):
        host.laser_state = LaserState()
    return host.laser_state


def page(m: ui.Menu, surf, rect, mouse) -> None:
    host = m.host
    ls = _get_laser(host)

    m.text(surf, "TARGETED OPTOGENETICS LASER", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "In-world aimable laser for real-time cellular stimulation and silencing without menus.",
           (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)

    # Scientific notice banner
    banner = pygame.Rect(rect.x + 24, rect.y + 70, rect.w - 48, 28)
    pygame.draw.rect(surf, (20, 32, 45), banner, border_radius=6)
    pygame.draw.rect(surf, (60, 140, 220), banner, 1, border_radius=6)
    m.text(surf, "TARGETED CELLULAR STIMULATION / SILENCING: Injects current directly into connectome rows. "
                 "Does not model Gal4/UAS drivers or opsin kinetics.",
           (banner.x + 10, banner.centery), (160, 210, 255), m.f_small, "midleft")

    y = rect.y + 110

    # Target cell type
    m.text(surf, "Target cell type:", (rect.x + 24, y + 4), ui.TEXT, m.f_text)
    m.text(surf, f"Active: {ls.target_type}", (rect.x + 180, y + 4), ui.INK, m.f_bold)

    insp = getattr(host, "inspect", None)
    if insp and insp.get("type"):
        it = insp["type"]
        m.button(surf, (rect.right - 240, y, 216, 28), f"Use Inspected ({it[:10]})",
                 lambda: ls.set_target(it), id="laser_use_insp", font=m.f_small)
    y += 36

    # Quick target chips
    chip_x = rect.x + 24
    for tgt in DEFAULT_TARGETS:
        tw = max(52, len(tgt) * 9 + 16)
        if chip_x + tw > rect.right - 24:
            chip_x = rect.x + 24
            y += 30
        on = ls.target_type.lower() == tgt.lower()
        r = pygame.Rect(chip_x, y, tw, 24)
        m.button(surf, r, tgt, (lambda t=tgt: ls.set_target(t)),
                 style="good" if on else "quiet", id=("chip", tgt), font=m.f_small)
        chip_x += tw + 8

    y += 44

    # Mode: Activate vs Silence
    m.text(surf, "Laser effect:", (rect.x + 24, y + 6), ui.TEXT, m.f_text)
    mode_idx = 0 if ls.mode == "activate" else 1
    m.segmented(surf, (rect.x + 180, y, 280, 32), ["Activate (+current)", "Silence (-current)"],
                mode_idx, lambda i: ls.set_mode("activate" if i == 0 else "silence"), id="laser_mode")
    col = (255, 160, 60) if ls.mode == "activate" else (80, 180, 255)
    cur_val = ls.current_value()
    m.text(surf, f"{cur_val:+.2f} pA/step", (rect.x + 480, y + 16), col, m.f_bold, "midleft")
    y += 44

    # Intensity slider
    m.text(surf, "Intensity:", (rect.x + 24, y + 6), ui.TEXT, m.f_text)
    m.slider(surf, (rect.x + 180, y, 280, 32), ls.intensity, 0.1, 3.0, 0.1, "{:.1f}x",
             lambda v: ls.set_intensity(v), lambda: None, id="laser_intensity",
             tip="Current intensity scale multiplier (0.1x to 3.0x)")
    y += 44

    # Trigger mode: Hold vs Pulse
    m.text(surf, "Trigger mode:", (rect.x + 24, y + 6), ui.TEXT, m.f_text)
    trig_idx = 0 if ls.trigger_mode == "hold" else 1
    m.segmented(surf, (rect.x + 180, y, 280, 32), ["Hold (continuous)", "Pulse (timed)"],
                trig_idx, lambda i: ls.set_trigger_mode("hold" if i == 0 else "pulse"), id="laser_trigger")
    y += 44

    if ls.trigger_mode == "pulse":
        m.text(surf, "Pulse duration:", (rect.x + 24, y + 6), ui.TEXT, m.f_text)
        m.slider(surf, (rect.x + 180, y, 280, 32), ls.pulse_duration * 1000, 50, 1000, 50, "{:.0f} ms",
                 lambda v: setattr(ls, "pulse_duration", v / 1000.0), lambda: None, id="laser_pulse_dur",
                 tip="Duration of single laser pulse upon trigger")
        y += 44

    # Status / In-World Equip Button
    y += 10
    has_laser_tool = False
    from kickthefly.game.kick_the_fly import TOOLS
    for idx, (tname, _, _) in enumerate(TOOLS):
        if tname == "laser":
            has_laser_tool = True
            is_equipped = getattr(host, "tool", None) == idx
            btn_txt = "Equipped in Hand" if is_equipped else "Equip Laser Tool in World"

            def equip(i=idx):
                host.tool = i
                m.back()

            m.button(surf, (rect.x + 24, y, 240, 42), btn_txt, equip,
                     style="primary" if not is_equipped else "good", id="laser_equip",
                     tip="Equips the laser tool in hand for immediate 2D or 3D in-world use.")
            break

    y += 56
    m.text(surf, "In-World Usage: Aim crosshair/cursor at the fly's body and click/hold. Beam shines cyan for "
                 "silencing and orange for activation.", (rect.x + 24, y), ui.LABEL, m.f_small)

    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("laser", "back"))
