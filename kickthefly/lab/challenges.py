"""Play mode challenges: three lab assays turned into games with a goal and a score.

  Teach it to pick the right door   T-maze olfactory conditioning (assays.tmaze_fly, the same steps and choice rule)
  How close can you sneak?          looming escape: the fly's giant fiber decides when it has seen you
  Find its sweet tooth              sugar response: the weakest sugar that still makes its proboscis motor neuron fire

The T-maze challenge trains the fly's real mushroom body synapses, then puts them back as they were when it ends, so
a practice round never changes how the fly treats your tools. Best scores are saved in scores.json in the data folder.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pygame

from kickthefly.lab import assays
from kickthefly.core import paths

INK, TEXT, LABEL, DIM = (240, 243, 248), (205, 212, 224), (130, 142, 160), (80, 88, 102)
AMBER, ACCENT, GOOD, BAD = (255, 176, 64), (86, 214, 255), (90, 200, 120), (230, 90, 80)
ODOR_COLORS = {"odor_a": (255, 150, 60), "odor_b": (180, 120, 255)}
ODOR_NAMES = {"odor_a": "orange smell", "odor_b": "purple smell"}

INFO = (
    ("tmaze", "Teach it to pick the right door",
     "Choose which door gives a zap. Train the fly, then watch it choose a door 10 times.", "right choices", "high"),
    ("sneak", "How close can you sneak?",
     "Creep up on the fly. Move fast and its escape neuron fires and it dodges. How close can you get?",
     "fly lengths away", "low"),
    ("sweet", "Find its sweet tooth",
     "Offer sugar at different strengths. Find the weakest sugar it still reaches for, in 8 tries.", "% sugar", "low"),
)


def scores_path():
    return paths.get().data_dir / "scores.json"


def load_scores() -> dict:
    try:
        return json.loads(scores_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def record_score(key: str, value: float, better: str) -> bool:
    """Save a score if it beats the best. Returns True for a new best."""
    s = load_scores()
    old = s.get(key)
    new_best = old is None or (value > old if better == "high" else value < old)
    if new_best:
        s[key] = value
        try:
            p = scores_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(s, indent=1), encoding="utf-8")
        except OSError:
            pass
    return new_best


def stars(key: str, value: float) -> int:
    if key == "tmaze":
        return 3 if value >= 9 else 2 if value >= 7 else 1 if value >= 5 else 0
    if key == "sneak":
        return 3 if value <= 1.0 else 2 if value <= 2.0 else 1 if value <= 3.5 else 0
    return 3 if value <= 10 else 2 if value <= 25 else 1 if value <= 50 else 0


def draw_stars(surf, center, n: int, size: int = 14) -> None:
    for i in range(3):
        cx = center[0] + (i - 1) * size * 2.4
        pts = []
        for k in range(10):
            r = size if k % 2 == 0 else size * 0.45
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((cx + r * math.cos(a), center[1] + r * math.sin(a)))
        pygame.draw.polygon(surf, AMBER if i < n else (60, 64, 76), pts)


class Button:
    def __init__(self, rect, label, action, enabled=True, style="normal"):
        self.rect, self.label, self.action, self.enabled, self.style = pygame.Rect(rect), label, action, enabled, style

    def draw(self, game, surf, mouse):
        over = self.enabled and self.rect.collidepoint(mouse)
        base = {"primary": (40, 110, 150), "danger": (130, 44, 44)}.get(self.style, (44, 50, 64))
        fill = tuple(min(255, c + 22) for c in base) if over else base
        if not self.enabled:
            fill = (32, 36, 46)
        pygame.draw.rect(surf, fill, self.rect, border_radius=8)
        if over:
            pygame.draw.rect(surf, (120, 132, 156), self.rect, 1, border_radius=8)
        game._text(surf, self.label, self.rect.center, INK if self.enabled else DIM, game.f_bold, "center")


class Challenge:
    key = ""
    overlay = True                    # a panel that blocks tool use

    def __init__(self, game):
        self.game = game
        self.buttons: list[Button] = []
        self.done = False

    @property
    def slot(self):
        return self.game.flies[self.game.focus]

    def update(self, now: float) -> None:
        pass

    def on_reaction(self, kind: str, slot) -> None:
        pass

    def click(self, pos) -> bool:
        for b in self.buttons:
            if b.enabled and b.rect.collidepoint(pos):
                self.game.sound.play("click")
                b.action()
                return True
        return self.overlay

    def end(self) -> None:
        self.game.challenge = None

    def panel(self, surf, title: str, subtitle: str, h: int = 560) -> pygame.Rect:
        from kickthefly.game.kick_the_fly import PLAY_W, H
        g = self.game
        panel = pygame.Rect(40, 30, min(PLAY_W - 80, 820), min(H - 60, h))
        veil = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        veil.fill((4, 5, 8, 140))
        surf.blit(veil, (0, 0))
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, (52, 60, 76), panel, 1, border_radius=16)
        g._text(surf, title.upper(), (panel.x + 22, panel.y + 14), INK, g.f_title)
        g._text(surf, subtitle, (panel.x + 24, panel.y + 58), LABEL, g.f_text)
        return panel

    def draw_buttons(self, surf, mouse) -> None:
        for b in self.buttons:
            b.draw(self.game, surf, mouse)


# --- Teach it to pick the right door ----------------------------------------------------------------------------------
class TMaze(Challenge):
    key = "tmaze"
    SPEED = 3.0
    CYCLES = 6
    TRIALS = 10

    @property
    def borrows_memory(self) -> bool:
        """While training, the fly's memory holds practice learning that must never be saved to disk."""
        return self.saved_memory is not None

    def __init__(self, game):
        super().__init__(game)
        self.phase = "choose"
        self.zap: str | None = None
        self.choices: list[tuple[str, bool]] = []
        self.anim = 0.0
        self.saved_memory = None
        self.cycle = 0
        self.until = 0
        self.drive: dict[str, float] = {}
        self.rng = np.random.default_rng(int(game.brain.seed) + 424242)

    @property
    def speed(self) -> float:
        return self.SPEED if self.phase in ("naive", "train", "test") else 1.0

    def _snapshot_memory(self) -> None:
        mem = self.game.brain.memory
        with mem.lock:
            self.saved_memory = (mem.w.copy(), {k: v.copy() for k, v in mem.templates.items()}, dict(mem.naive_mbon),
                                 {k: list(v) for k, v in mem.log.items()}, mem.dirty)

    def _restore_memory(self) -> None:
        if self.saved_memory is None:
            return
        br = self.game.brain
        mem = br.memory
        w, templates, naive, log, dirty = self.saved_memory
        with br.step_lock, mem.lock:
            mem.w[:] = w
            mem.templates, mem.naive_mbon, mem.log, mem.dirty = templates, naive, log, dirty
            mem._write_back()
        self.saved_memory = None

    def pick(self, door: str) -> None:
        br = self.game.brain
        if br.memory is None:
            return
        self.zap = door
        self._snapshot_memory()
        self.phase, self.stage = "naive", 0
        self.until = br.steps + 240

    def end(self) -> None:
        self._restore_memory()
        super().end()

    def update(self, now: float) -> None:
        br = self.game.brain
        if self.phase in ("choose", "done") or br.memory is None:
            return
        plus = self.zap
        minus = "odor_b" if plus == "odor_a" else "odor_a"
        step = br.steps
        mem = br.memory
        if self.phase == "naive":                       # the fly smells each door once so it knows the smells
            odor = plus if self.stage == 0 else minus
            br.poke("scent", odor, 0.5)
            if self.game.frame % 3 == 0:
                mem.observe(odor, br.sim.activity.rates())
            if step >= self.until:
                self.stage += 1
                self.until = step + 240
                if self.stage == 2:
                    self.phase, self.stage, self.cycle = "train", 0, 0
                    self.until = step + 240
            return
        if self.phase == "train":                       # the same schedule as assays.tmaze_fly
            schedule = ((plus, False, 240), (plus, True, 200), (None, False, 260), (minus, False, 440), (None, False, 260))
            odor, shock, _ = schedule[self.stage]
            if odor:
                br.poke("scent", odor, 0.5)
                if self.game.frame % 3 == 0:
                    mem.observe(odor, br.sim.activity.rates())
            if shock:
                br.poke("punish", None, 1.0)
                br.poke("legs", "L", 0.6)
                br.poke("legs", "R", 0.6)
            if step >= self.until:
                self.stage = (self.stage + 1) % len(schedule)
                if self.stage == 0:
                    self.cycle += 1
                    if self.cycle >= self.CYCLES:
                        self.phase, self.stage, self.choices = "test", 0, []
                self.until = step + schedule[self.stage][2]
            return
        if self.phase == "test":
            order = (plus, minus) if len(self.choices) % 2 == 0 else (minus, plus)
            if self.stage < 2:                          # smell each door
                odor = order[self.stage]
                br.poke("scent", odor, 0.5)
                if self.game.frame % 3 == 0:
                    mem.observe(odor, br.sim.activity.rates())
                if step >= self.until:
                    fear, like = mem.memory_of(odor)
                    self.drive[odor] = like - fear
                    self.stage += 1
                    self.until = step + (60 if self.stage < 2 else 1)
            elif self.stage == 2:                       # choose (assays.tmaze_fly's rule)
                noise = 0.08
                pick_plus = (self.drive[plus] + self.rng.normal(0, noise) > self.drive[minus] + self.rng.normal(0, noise))
                door = plus if pick_plus else minus
                self.choices.append((door, not pick_plus))
                self.anim, self.stage = now, 3
                self.game.sound.play("pop" if not pick_plus else "zap", 0.5)
            elif now - self.anim > 0.9:
                if len(self.choices) >= self.TRIALS:
                    self.phase = "done"
                    right = sum(ok for _, ok in self.choices)
                    self.best = record_score("tmaze", right, "high")
                    self._restore_memory()
                else:
                    self.stage, self.until = 0, step + 60

    def draw(self, surf, now, mouse) -> None:
        g = self.game
        p = self.panel(surf, "Teach it to pick the right door", "One door zaps. Can the fly learn which one?", 580)
        self.buttons = []
        cx = p.centerx
        doors = {"odor_a": pygame.Rect(p.x + 120, p.y + 120, 150, 210), "odor_b": pygame.Rect(p.right - 270, p.y + 120, 150, 210)}
        for odor, r in doors.items():
            col = ODOR_COLORS[odor]
            pygame.draw.rect(surf, tuple(c // 3 for c in col), r, border_radius=12)
            pygame.draw.rect(surf, col, r, 3, border_radius=12)
            for k in range(3):                               # wavy smell lines
                pts = [(r.centerx - 30 + x, r.y + 40 + k * 22 + 5 * math.sin(now * 3 + x / 8 + k)) for x in range(0, 61, 6)]
                pygame.draw.lines(surf, col, False, pts, 2)
            g._text(surf, ODOR_NAMES[odor], (r.centerx, r.bottom + 8), col, g.f_bold, "midtop")
            if self.zap == odor:
                g._text(surf, "ZAP", (r.centerx, r.bottom - 40), AMBER, g.f_head, "center")
        # the fly icon at the choice point, walking to its latest choice
        fx, fy = cx, p.y + 330
        if self.phase == "test" and self.stage == 3 and self.choices:
            door = doors[self.choices[-1][0]]
            e = min(1.0, (now - self.anim) / 0.7)
            fx, fy = fx + (door.centerx - fx) * e, fy + (door.bottom - 20 - fy) * e
        pygame.draw.ellipse(surf, (70, 60, 40), (fx - 14, fy - 9, 28, 18))
        pygame.draw.circle(surf, (170, 40, 40), (int(fx + 12), int(fy - 2)), 5)
        y = p.y + 390
        if self.phase == "choose":
            g._text(surf, "Pick the door that zaps the fly:", (cx, y), TEXT, g.f_text, "midtop")
            mem_ok = g.brain.memory is not None
            self.buttons = [Button((p.x + 120, y + 34, 150, 44), "Orange zaps", lambda: self.pick("odor_a"), mem_ok),
                            Button((p.right - 270, y + 34, 150, 44), "Purple zaps", lambda: self.pick("odor_b"), mem_ok)]
        elif self.phase in ("naive", "train"):
            total = self.CYCLES
            frac = (self.cycle + self.stage / 5) / total if self.phase == "train" else 0.0
            label = "Letting it smell both doors..." if self.phase == "naive" else \
                f"Training {self.cycle + 1}/{total}: " + ("smell + zap" if self.stage == 1 else "smell" if self.stage in (0, 3) else "rest")
            g._text(surf, label, (cx, y), TEXT, g.f_text, "midtop")
            pygame.draw.rect(surf, (30, 36, 48), (p.x + 60, y + 32, p.w - 120, 12), border_radius=6)
            pygame.draw.rect(surf, AMBER, (p.x + 60, y + 32, max(10, int((p.w - 120) * frac)), 12), border_radius=6)
        elif self.phase in ("test", "done"):
            right = sum(ok for _, ok in self.choices)
            g._text(surf, f"Choices: {len(self.choices)}/{self.TRIALS}    right door: {right}", (cx, y), INK, g.f_bold, "midtop")
            for i, (_, ok) in enumerate(self.choices):
                pygame.draw.circle(surf, GOOD if ok else BAD, (cx - 9 * 22 // 2 + i * 22, y + 40), 8)
            if self.phase == "done":
                draw_stars(surf, (cx, y + 84), stars("tmaze", right))
                msg = "New best!" if getattr(self, "best", False) else f"Best: {load_scores().get('tmaze', right):.0f}/10"
                g._text(surf, msg, (cx, y + 106), AMBER, g.f_text, "midtop")
                self.buttons.append(Button((cx - 200, p.bottom - 64, 180, 44), "Play again", lambda: self.game.start_challenge("tmaze"),
                                           style="primary"))
        close = "Close" if self.phase in ("choose", "done") else "Stop"
        self.buttons.append(Button((p.right - 160 if self.phase != "done" else cx + 20, p.bottom - 64, 140, 44), close, self.end))
        self.draw_buttons(surf, mouse)


# --- How close can you sneak? -----------------------------------------------------------------------------------------
class Sneak(Challenge):
    key = "sneak"
    overlay = False

    def __init__(self, game):
        super().__init__(game)
        self.closest = math.inf
        self.state = "armed"               # armed -> sneaking -> result
        self.result: tuple[str, float] | None = None
        self.result_t = 0.0
        self.hits0 = game.flies[game.focus].hits

    def update(self, now: float) -> None:
        g = self.game
        slot = self.slot
        d = g.sneak_distance(slot)
        airborne = now < slot.fly.escape_until              # (fly.flying stays true after its first flight)
        if self.state == "result":
            if now - self.result_t > 2.5 and d > 3.0 and not airborne and not slot.fly.dead:
                self.state, self.closest, self.hits0 = "armed", math.inf, slot.hits
            return
        if slot.fly.dead:
            return
        if self.state == "armed" and d < 3.0 and not airborne:
            self.state = "sneaking"
        if self.state == "sneaking":
            self.closest = min(self.closest, d)
            if slot.hits > self.hits0 or d < 0.35:
                self._finish("caught", 0.0 if d < 0.35 else self.closest, now)

    def on_reaction(self, kind: str, slot) -> None:
        if kind == "DODGE" and slot is self.slot and self.state == "sneaking":
            self._finish("seen", self.closest, self.game.clock.now)

    def _finish(self, how: str, dist: float, now: float) -> None:
        self.state, self.result, self.result_t = "result", (how, dist), now
        self.best = record_score("sneak", round(dist, 2), "low")
        self.game.sound.play("dodge" if how == "seen" else "yum", 0.6)

    def draw(self, surf, now, mouse) -> None:
        from kickthefly.game.kick_the_fly import PLAY_W
        g = self.game
        w, h = 330, 120
        box = pygame.Rect(PLAY_W // 2 - w // 2, 120, w, h)
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 200), card.get_rect(), border_radius=12)
        surf.blit(card, box)
        g._text(surf, "HOW CLOSE CAN YOU SNEAK?", (box.x + 14, box.y + 8), AMBER, g.f_bold)
        best = load_scores().get("sneak")
        self.buttons = [Button((box.right - 70, box.y + 6, 60, 26), "Quit", self.end)]
        if self.state == "armed":
            g._text(surf, "Back off, then creep toward the fly.", (box.x + 14, box.y + 38), TEXT, g.f_text)
        elif self.state == "sneaking":
            d = g.sneak_distance(self.slot)
            g._text(surf, f"{d:4.1f} fly lengths away", (box.x + 14, box.y + 36), INK, g.f_head)
            g._text(surf, f"closest {self.closest:.1f}", (box.x + 14, box.y + 66), TEXT, g.f_text)
        elif self.result:
            how, dist = self.result
            msg = "You touched it before it saw you!" if how == "caught" else f"It saw you at {dist:.1f} fly lengths"
            g._text(surf, msg, (box.x + 14, box.y + 36), GOOD if how == "caught" else INK, g.f_bold)
            draw_stars(surf, (box.x + 70, box.y + 76), stars("sneak", dist), 10)
            if getattr(self, "best", False):
                g._text(surf, "New best!", (box.x + 140, box.y + 68), AMBER, g.f_text)
        if best is not None:
            g._text(surf, f"best {best:.1f}", (box.right - 14, box.bottom - 24), LABEL, g.f_small, "topright")
        self.draw_buttons(surf, mouse)


# --- Find its sweet tooth ---------------------------------------------------------------------------------------------
class Sweet(Challenge):
    key = "sweet"
    TRIES = 8

    def __init__(self, game):
        super().__init__(game)
        self.dose = 50
        self.phase = "ready"
        self.offers: list[tuple[int, bool, float]] = []
        self.samples: dict[str, list] = {"before": [], "during": []}
        self.until = 0
        self.dragging = False

    def offer(self) -> None:
        br = self.game.brain
        self.phase, self.samples = "before", {"before": [], "during": []}
        self.until = br.steps + 200

    def update(self, now: float) -> None:
        br = self.game.brain
        if self.phase == "before":
            self.samples["before"].append(br.hz("proboscis"))
            if br.steps >= self.until:
                self.phase, self.until = "during", br.steps + 200
        elif self.phase == "during":
            if self.dose > 0:
                br.poke("sweet", None, 0.5, recruit=self.dose / 100)
            self.samples["during"].append(br.hz("proboscis"))
            if br.steps >= self.until:
                b = float(np.mean(self.samples["before"])) if self.samples["before"] else 0.0
                d = float(np.mean(self.samples["during"])) if self.samples["during"] else 0.0
                ratio = d / max(b, 1.0)
                ext = ratio >= assays.PER_RATIO
                self.offers.append((self.dose, ext, ratio))
                self.phase, self.shown = "shown", now
                self.game.sound.play("yum" if ext else "click", 0.6)
                if ext:
                    self.game.on_reaction("PROBOSCIS", self.slot)
                    self.game.note(f"PROBOSCIS MN9 x{ratio:.1f} to {self.dose}% sugar")
        elif self.phase == "shown" and now - self.shown > 1.4:
            if len(self.offers) >= self.TRIES:
                self.phase = "done"
                good = [d for d, ext, _ in self.offers if ext]
                self.score = min(good) if good else None
                self.best = self.score is not None and record_score("sweet", self.score, "low")
            else:
                self.phase = "ready"

    def click(self, pos) -> bool:
        if self.phase == "ready" and self.track.inflate(0, 24).collidepoint(pos):
            frac = (pos[0] - self.track.x) / self.track.w
            self.dose = int(round(np.clip(frac, 0, 1) * 20)) * 5
            return True
        return super().click(pos)

    def draw(self, surf, now, mouse) -> None:
        g = self.game
        p = self.panel(surf, "Find its sweet tooth", "Offer sugar. Find the weakest sugar it still reaches for.", 520)
        cx = p.centerx
        self.buttons = []
        g._text(surf, f"Sugar strength: {self.dose}%", (p.x + 60, p.y + 110), INK, g.f_head)
        self.track = pygame.Rect(p.x + 60, p.y + 160, p.w - 120, 8)
        pygame.draw.rect(surf, (50, 56, 68), self.track, border_radius=4)
        pygame.draw.rect(surf, (255, 150, 190), (self.track.x, self.track.y, int(self.track.w * self.dose / 100), 8), border_radius=4)
        pygame.draw.circle(surf, INK, (self.track.x + int(self.track.w * self.dose / 100), self.track.centery), 10)
        g._text(surf, "click the bar to set the strength", (p.x + 60, p.y + 178), DIM, g.f_small)
        # proboscis icon
        head = (cx, p.y + 270)
        pygame.draw.circle(surf, (120, 90, 50), head, 34)
        pygame.draw.circle(surf, (190, 40, 40), (head[0] - 18, head[1] - 10), 9)
        pygame.draw.circle(surf, (190, 40, 40), (head[0] + 18, head[1] - 10), 9)
        ext = self.phase == "shown" and self.offers and self.offers[-1][1]
        length = 46 if ext else 10
        pygame.draw.line(surf, (150, 110, 70), (head[0], head[1] + 26), (head[0], head[1] + 26 + length), 8)
        if self.phase in ("before", "during"):
            g._text(surf, "tasting..." if self.phase == "during" else "watching...", (cx, p.y + 350), TEXT, g.f_text, "midtop")
        elif self.phase == "shown":
            d, e, r = self.offers[-1]
            g._text(surf, "It reached for it!" if e else "Not interested.", (cx, p.y + 350), GOOD if e else LABEL, g.f_bold, "midtop")
        row = "  ".join(f"{d}%{'+' if e else '-'}" for d, e, _ in self.offers)
        g._text(surf, f"Tries {len(self.offers)}/{self.TRIES}:  {row}", (p.x + 30, p.bottom - 110), TEXT, g.f_text)
        if self.phase == "done":
            if self.score is None:
                g._text(surf, "It never reached for the sugar this time.", (cx, p.y + 384), INK, g.f_bold, "midtop")
            else:
                g._text(surf, f"Weakest sugar it reached for: {self.score}%", (cx, p.y + 384), INK, g.f_bold, "midtop")
                draw_stars(surf, (cx, p.y + 424), stars("sweet", self.score))
            self.buttons.append(Button((cx - 200, p.bottom - 64, 180, 44), "Play again",
                                       lambda: self.game.start_challenge("sweet"), style="primary"))
            self.buttons.append(Button((cx + 20, p.bottom - 64, 140, 44), "Close", self.end))
        else:
            self.buttons.append(Button((cx - 90, p.bottom - 64, 180, 44), "Offer sugar", self.offer,
                                       enabled=self.phase == "ready", style="primary"))
            self.buttons.append(Button((p.right - 160, p.bottom - 64, 140, 44), "Close", self.end))
        best = load_scores().get("sweet")
        if best is not None:
            g._text(surf, f"best: {best}%", (p.right - 30, p.y + 110), LABEL, g.f_text, "topright")
        self.draw_buttons(surf, mouse)


CLASSES = {"tmaze": TMaze, "sneak": Sneak, "sweet": Sweet}


def page_challenges(m, surf, rect, mouse) -> None:
    from kickthefly.ui import menu as ui

    game = m.host
    m.text(surf, "CHALLENGES", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.text(surf, "Games built on real fly experiments. The fly's brain decides how it does.", (rect.x + 24, rect.y + 50),
           ui.LABEL, m.f_small)
    scores = load_scores()
    y = rect.y + 96
    for key, title, desc, unit, better in INFO:
        card = pygame.Rect(rect.x + 24, y, rect.w - 48, 150)
        pygame.draw.rect(surf, (28, 32, 42), card, border_radius=12)
        m.text(surf, title, (card.x + 20, card.y + 16), ui.INK, m.f_head)
        m.text(surf, desc, (card.x + 20, card.y + 54), ui.TEXT, m.f_text)
        best = scores.get(key)
        if best is not None:
            shown = f"{best:.0f}/10" if key == "tmaze" else f"{best:.1f} fly lengths" if key == "sneak" else f"{best:.0f}%"
            m.text(surf, f"Best: {shown}", (card.x + 20, card.y + 96), ui.AMBER, m.f_bold)
            draw_stars(surf, (card.x + 250, card.y + 106), stars(key, best), 11)
        else:
            m.text(surf, "Not played yet", (card.x + 20, card.y + 96), ui.LABEL, m.f_text)
        m.button(surf, (card.right - 170, card.y + 86, 150, 46), "Start", (lambda k=key: game.start_challenge(k)),
                 style="primary", id=("challenge", key))
        y += 166
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("ch", "back"))
