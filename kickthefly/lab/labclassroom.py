"""Classroom mode and lecture presentation UI screen.

Sequential step-by-step walkthroughs of verified Drosophila neural circuits
with Next/Back controls, step explanations, participating neuron breakdowns,
scientific citations, and interactive live demonstration actions.
"""
from __future__ import annotations

import pygame

from kickthefly.ui import menu as ui
from kickthefly.lab.classroom import CURATED_LECTURES, ClassroomSession


def _get_session(host) -> ClassroomSession:
    if not hasattr(host, "classroom_session") or host.classroom_session is None:
        br = getattr(host, "brain", None)
        host.classroom_session = ClassroomSession(lecture_id="looming", brain=br)
    elif getattr(host, "brain", None) is not None:
        host.classroom_session.brain = host.brain
    return host.classroom_session


def page(m: ui.Menu, surf, rect: pygame.Rect, mouse) -> None:
    host = m.host
    sess = _get_session(host)
    proto = sess.protocol
    step = sess.current_step

    # Header
    m.text(surf, "CLASSROOM MODE & LECTURE PROTOCOLS", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Curated sequential demonstrations for lectures and teaching. "
                 "Every behavior is either grounded in connectome wiring or labelled a game rule.",
           (rect.x + 24, rect.y + 48), ui.LABEL, m.f_small)

    y = rect.y + 76

    # Lecture selection tabs across top
    m.text(surf, "Select Lecture:", (rect.x + 24, y + 4), ui.TEXT, m.f_small)
    tx = rect.x + 130
    lectures_meta = [
        ("looming", "Looming escape"),
        ("tmaze", "T-maze conditioning"),
        ("moonwalker", "Moonwalker (MDN)"),
        ("sugar", "Sugar feeding"),
        ("gf_lesion", "GF lesion"),
    ]
    for lid, ltitle in lectures_meta:
        active = (sess.lecture_id == lid)
        tw = max(80, len(ltitle) * 8 + 16)
        r_tab = pygame.Rect(tx, y, tw, 26)
        if r_tab.right > rect.right - 24:
            break
        bg_col = (40, 90, 160) if active else (24, 28, 38)
        border_col = (100, 180, 255) if active else (50, 58, 76)
        pygame.draw.rect(surf, bg_col, r_tab, border_radius=5)
        pygame.draw.rect(surf, border_col, r_tab, 1, border_radius=5)
        m.text(surf, ltitle, r_tab.center, (255, 255, 255) if active else ui.TEXT, m.f_small, "center")
        m.button(surf, r_tab, "", lambda id_=lid: sess.set_lecture(id_), id=("class_tab", lid))
        tx += tw + 8

    y += 36

    # Protocol title & Step indicator banner
    banner = pygame.Rect(rect.x + 24, y, rect.w - 48, 32)
    pygame.draw.rect(surf, (20, 24, 34), banner, border_radius=6)
    pygame.draw.rect(surf, (60, 70, 95), banner, 1, border_radius=6)

    m.text(surf, f"Lecture: {proto.title}", (banner.x + 12, banner.centery), (140, 190, 255), m.f_bold, "midleft")
    step_str = f"Step {sess.step_idx + 1} of {sess.total_steps}"
    m.text(surf, step_str, (banner.right - 12, banner.centery), ui.AMBER, m.f_bold, "midright")

    y += 42

    # Step Card
    card_w = rect.w - 48
    card_h = rect.h - y - 72
    card = pygame.Rect(rect.x + 24, y, card_w, card_h)
    pygame.draw.rect(surf, (18, 22, 32), card, border_radius=8)
    pygame.draw.rect(surf, (40, 48, 66), card, 1, border_radius=8)

    # Step Title
    m.text(surf, step.title, (card.x + 18, card.y + 16), ui.INK, m.f_head)

    # Grounding tag
    tag_x = card.right - 18
    grounding_text = step.grounding
    m.text(surf, grounding_text, (tag_x, card.y + 20), (100, 200, 140), m.f_small, "topright")

    content_y = card.y + 54

    # Left Column: Narrative Explanation & Key Takeaway & Live Action (w = ~60% card)
    left_w = int(card.w * 0.58)
    m.text(surf, "CIRCUIT MECHANISM & EXPLANATION", (card.x + 18, content_y), ui.LABEL, m.f_small)
    content_y += 22

    # Wrap explanation text
    m.wrapped(surf, step.explanation, (card.x + 18, content_y), left_w - 24, ui.TEXT, m.f_text, max_lines=7)
    content_y += 120

    # Key Takeaway Box
    takeaway_rect = pygame.Rect(card.x + 18, content_y, left_w - 24, 46)
    pygame.draw.rect(surf, (28, 38, 52), takeaway_rect, border_radius=6)
    pygame.draw.rect(surf, (70, 120, 180), takeaway_rect, 1, border_radius=6)
    m.text(surf, "Key Takeaway:", (takeaway_rect.x + 10, takeaway_rect.y + 8), (140, 200, 255), m.f_small)
    m.wrapped(surf, step.key_takeaway, (takeaway_rect.x + 10, takeaway_rect.y + 24), takeaway_rect.w - 20, ui.INK, m.f_small, max_lines=2)

    content_y += 56

    # Action demonstration button
    if step.action:
        act_desc = step.action.get("kind", "demo").upper()
        target_name = step.action.get("target", "")
        btn_label = f"▶ Demonstrate Step ({act_desc} {target_name})".strip()
        m.button(
            surf,
            (card.x + 18, content_y, 280, 36),
            btn_label,
            lambda: sess.execute_step_action(getattr(host, "brain", None), host),
            id="class_run_demo",
            style="primary",
        )
        if sess.last_action_applied:
            m.text(surf, f"Status: {sess.last_action_applied}", (card.x + 310, content_y + 18), (120, 220, 150), m.f_small, "midleft")
    else:
        m.text(surf, "No active stimulus required for this step.", (card.x + 18, content_y + 10), ui.LABEL, m.f_small)

    # Right Column: Neurons Involved & Scientific Citation (w = ~40% card)
    right_x = card.x + left_w + 12
    right_w = card.w - left_w - 30

    m.text(surf, "NEURONS INVOLVED", (right_x, card.y + 54), ui.LABEL, m.f_small)
    ny = card.y + 76
    for nr in step.neurons[:4]:
        n_box = pygame.Rect(right_x, ny, right_w, 42)
        pygame.draw.rect(surf, (24, 28, 40), n_box, border_radius=5)
        pygame.draw.rect(surf, (45, 54, 76), n_box, 1, border_radius=5)
        m.text(surf, nr.label, (n_box.x + 8, n_box.y + 6), (255, 200, 110), m.f_bold)
        if nr.count > 0:
            m.text(surf, f"n={nr.count}", (n_box.right - 8, n_box.y + 6), ui.LABEL, m.f_small, "topright")
        m.wrapped(surf, nr.role, (n_box.x + 8, n_box.y + 22), n_box.w - 16, ui.TEXT, m.f_small, max_lines=1)
        ny += 48

    ny += 8
    # Scientific Citation
    m.text(surf, "SCIENTIFIC CITATION", (right_x, ny), ui.LABEL, m.f_small)
    ny += 22
    cite_box = pygame.Rect(right_x, ny, right_w, 54)
    pygame.draw.rect(surf, (15, 20, 30), cite_box, border_radius=6)
    pygame.draw.rect(surf, (55, 75, 105), cite_box, 1, border_radius=6)
    m.wrapped(surf, step.citation, (cite_box.x + 10, cite_box.y + 8), cite_box.w - 20, (180, 210, 255), m.f_small, max_lines=3)

    # Bottom Control Bar
    ctrl_y = rect.bottom - 58

    # Reset button
    m.button(surf, (rect.x + 24, ctrl_y, 140, 42), "↺ Reset Protocol", sess.reset, id="class_reset")

    # Prev button (disabled on step 0)
    can_prev = sess.step_idx > 0
    m.button(
        surf,
        (rect.x + 180, ctrl_y, 150, 42),
        "⟵ Previous Step",
        (sess.prev_step if can_prev else lambda: None),
        id="class_prev",
        enabled=can_prev,
    )

    # Next button (disabled on last step)
    can_next = sess.step_idx < sess.total_steps - 1
    m.button(
        surf,
        (rect.x + 344, ctrl_y, 150, 42),
        "Next Step ⟶",
        (sess.next_step if can_next else lambda: None),
        id="class_next",
        style=("primary" if can_next else "normal"),
        enabled=can_next,
    )

    # Back to Lab Hub button
    m.button(surf, (rect.right - 164, ctrl_y, 140, 42), "Back", m.back, style="primary", id="class_back")
