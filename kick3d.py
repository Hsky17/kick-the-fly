"""Kick the Fly in first person: the same live connectome fly, in a 3D room.

You walk around a living room (WASD, mouse to look) holding the selected tool. Everything the brain does is
unchanged from the 2D game (kick_the_fly.py): the same touch, heat, cold, smell, taste, wind, humidity, light and
looming inputs, the same descending-neuron reactions, pain, reward, learning, surgery and autopsy. Only the world
is new: the ragdoll, the fly's walking, turning and flight now happen in 3D, and tools are aimed with the camera.

Scale: the fly is a cartoon about 60 cm long. The 2D game's physics in pixels map to meters by S = 0.006, so every
tuned speed, impulse and threshold carries over.

Looming in 3D: the threats the fly sees are your body, the tool in your hand, a swatter mid-swing, thrown bombs
and the spider. Charge at it or swing fast and its giant fiber makes it dodge; walk up slowly and it won't notice.

    .venv\\Scripts\\python.exe kick_the_fly.py            (3D, the default)
    .venv\\Scripts\\python.exe kick_the_fly.py --2d       (the original 2D game)
"""
from __future__ import annotations

import math
import random

import threading
import time
from pathlib import Path

import moderngl
import numpy as np
import pygame

import kick_the_fly as k2
from kick_the_fly import (ABD, FOOT, HEAD, KNEE, LINKS, MAX_HEALTH, N_P, PULL, RADIUS, REST, THRESH, THX, TOOLS,
                          TORCH_KEYS, TRIPOD, WING)
from render3d import (P_BOOKS, P_CEIL, P_EYE, P_ICE, P_NONE, P_PAPER, P_RUG, P_STRIPES, P_WALLPAPER, P_WATER,
                      P_WOOD, Renderer, frame_from_x, look_at, perspective, rot_x, rot_y, rot_z, segment, trs)

S = 0.006                                   # meters per 2D pixel
RX, RY, RZ = 4.2, 3.0, 3.6                  # room half-width, height, half-depth
GRAV = 0.9 * S
STAND3 = k2.STAND * S
WATER3 = 0.6
EYE, CROUCH_EYE = 1.6, 0.85
LAMP3 = np.array([0.0, 2.2, -0.8])
FAN3 = np.array([-RX + 0.45, 0.0, 0.6])
PAPER3 = (-1.6, 1.6, -1.9, 1.1)             # flypaper x0, x1, z0, z1
REACH, GRAB_REACH = 3.0, 3.2
PANEL_W = k2.W - k2.PLAY_W                 # the brain panel's width in HUD units (390)
PANEL_MODES = (("solid", 255), ("see-through", 150), ("faint", 70), ("hidden", 0))
UI_MODES = ("crisp", "large")


def compute_layout(Wn: int, Hn: int, ui_mode: str, panel_mode: int):
    """HUD units -> window pixels. The HUD always fills the window (no black bars). In "crisp" mode the scale is a
    whole number whenever the window is at least 700 px tall per step (1440p -> 2x), so text maps to whole pixels;
    "large" keeps the original 760-unit-tall HUD, smoothly scaled. Returns (scale, hud_w, hud_h, play_w, view_w):
    play_w is the HUD area left of the brain panel, view_w how wide the 3D view is."""
    s = max(1, Hn // 700) if ui_mode == "crisp" and Hn >= 700 else Hn / 760
    need = 1280 if PANEL_MODES[panel_mode][0] != "hidden" else 900
    if Wn / s < need:                               # too narrow for the HUD: scale down to fit the width
        s = Wn / need
    hud_w, hud_h = max(1, int(round(Wn / s))), max(1, int(round(Hn / s)))
    hidden = PANEL_MODES[panel_mode][0] == "hidden"
    play_w = hud_w if hidden else hud_w - PANEL_W
    view_w = play_w if PANEL_MODES[panel_mode][0] == "solid" else hud_w
    return s, hud_w, hud_h, play_w, view_w
TOOL_SIZE = {"flick": 0.06, "swatter": 0.2, "bomb": 0.08, "torch": 0.09, "cleaner": 0.09, "zapper": 0.12,
             "freeze": 0.09, "spider": 0.08}
SKY_CLEAR = (0.08, 0.09, 0.11)

# furniture you and the fly bump into: (min corner, max corner)
COLLIDERS = [
    (np.array([-3.5, 0.0, -RZ]), np.array([-1.1, 0.62, -RZ + 1.0])),        # couch
    (np.array([RX - 0.45, 0.0, -1.9]), np.array([RX, 2.05, -0.1])),         # bookshelf
    (np.array([RX - 1.0, 0.0, RZ - 1.0]), np.array([RX - 0.2, 0.55, RZ - 0.2])),   # plant
]

HELP3D = (
    ("WASD", "walk (Shift sprint, Ctrl crouch)"),
    ("Mouse", "look; left click uses the tool in your hand"),
    ("1-9, 0 / wheel", "pick a tool"),
    ("Tab", "free the mouse to click the brain panel and menus"),
    ("B", "big live brain view; click a neuron to inspect it"),
    ("O", "brain surgery"),
    ("E", "arena: room, fan, flypaper, pool, lamp"),
    ("P / I", "pain neurons / immortal mode"),
    ("M", "mute"),
    ("F12 / G", "save a screenshot / a GIF of the last 6 s"),
    ("V", "brain panel: solid, see-through, faint, hidden"),
    ("U", "menu size: crisp (whole-pixel scaling) or large"),
    ("F11", "fullscreen"),
    ("R", "new fly"),
    ("Esc", "free the mouse, close menus, then quit"),
)


def _rest3() -> np.ndarray:
    """Rest pose in the fly's body frame (forward, up, side), meters; legs splay to both sides."""
    R = np.zeros((N_P, 3))
    R[:, 0] = REST[:, 0] * S
    R[:, 1] = -REST[:, 1] * S
    for i in range(3):
        R[KNEE[i + 3], :2] = R[KNEE[i], :2]
        R[FOOT[i + 3], :2] = R[FOOT[i], :2]
        R[KNEE[i], 2], R[KNEE[i + 3], 2] = 0.13, -0.13
        R[FOOT[i], 2], R[FOOT[i + 3], 2] = 0.25, -0.25
    R[WING[1], :2] = R[WING[0], :2]
    R[WING[0], 2], R[WING[1], 2] = 0.07, -0.07
    return R


REST3 = _rest3()
RAD3 = RADIUS * S
LINK_LEN3 = [float(np.linalg.norm(REST3[a] - REST3[b])) for a, b, _, _ in LINKS]


def body_axes(yaw: float):
    fwd = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    side = np.array([-math.sin(yaw), 0.0, math.cos(yaw)])
    return fwd, np.array([0.0, 1.0, 0.0]), side


def to_world(off: np.ndarray, yaw: float) -> np.ndarray:
    f, u, s = body_axes(yaw)
    return np.outer(off[:, 0], f) + np.outer(off[:, 1], u) + np.outer(off[:, 2], s)


def angle_to(v) -> float:
    return math.atan2(float(v[2]), float(v[0]))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def push_out_boxes(p: np.ndarray, r) -> None:
    """Push points (n, 3) out of the furniture boxes along the shallowest axis."""
    for lo, hi in COLLIDERS:
        inside = np.all((p > lo - r[:, None]) & (p < hi + r[:, None]), axis=1) if np.ndim(r) else \
            np.all((p > lo - r) & (p < hi + r), axis=1)
        for i in np.flatnonzero(inside):
            ri = r[i] if np.ndim(r) else r
            pen = np.concatenate([p[i] - (lo - ri), (hi + ri) - p[i]])        # distance to each face
            k = int(np.argmin(pen))
            if k < 3:
                p[i, k] = lo[k] - ri
            else:
                p[i, k - 3] = hi[k - 3] + ri


class Fly3D:
    """The 2D ragdoll lifted into 3D: the same particles, links, pose pull and flight, facing any direction."""

    def __init__(self, pos_xz=(0.0, -0.8), yaw: float = math.pi / 2):
        self.yaw = self.yaw_target = yaw
        self.p = to_world(REST3, yaw) + (pos_xz[0], STAND3, pos_xz[1])
        self.prev = self.p.copy()
        self.anchor = np.array(pos_xz, float)
        self.phase = 0.0
        self.grabbed: int | None = None
        self.stun_until = self.escape_until = self.walk_until = self.back_until = self.flail_until = 0.0
        self.run = False
        self.turn_ready = self.escape_ready = 0.0
        self.recover, self.hurt = 1.0, 0.0
        self.last_hit = self.p[THX].copy()
        self.action = "idle"
        self.health = MAX_HEALTH
        self.dead_at: float | None = None
        self.char = self.soak = self.melt = self.frost = self.venom = 0.0
        self.burn_until = self.zap_until = self.eating_until = 0.0
        self.dissolved_at = self.frozen_at = self.shattered_at = None
        self.hover = self.p[THX].copy()
        self.fly_target = self.hover.copy()
        self.wander = False
        self.power = 1.0
        self.wrapped = False
        self.arena = "room"
        self.wind = np.zeros(3)
        self.wet = 0.0
        self.stuck: dict[int, np.ndarray] = {}

    @property
    def dead(self) -> bool:
        return self.dead_at is not None

    @property
    def flying(self) -> bool:
        return (self.escape_until > 0 and self.grabbed is None and not self.dead and not self.wrapped and self.melt < 0.3
                and self.frost < 0.5 and self.venom < 0.5 and self.wet <= 0 and len(self.stuck) < 2)

    def nearest_to_ray(self, eye, d, reach: float, max_perp: float):
        rel = self.p - eye
        t = rel @ d
        perp = np.linalg.norm(rel - np.outer(t, d), axis=1) - RAD3
        ok = (t > 0) & (t < reach) & (perp < max_perp)
        if not ok.any():
            return None, None, None
        i = int(np.flatnonzero(ok)[np.argmin(perp[ok])])
        return i, float(t[i]), float(perp[i])

    def impulse(self, i: int, v) -> None:
        self.prev[i] -= v

    def stun(self, now: float, s: float) -> None:
        self.stun_until = max(self.stun_until, now + s)
        self.hurt = 1.0
        self.escape_until = min(self.escape_until, now)

    def away_from(self, point) -> np.ndarray:
        v = self.p[THX] - np.asarray(point, float)
        v[1] = 0
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else body_axes(self.yaw)[0]

    def escape(self, now: float, seconds: float = 2.4, wander: bool = False) -> None:
        away = self.away_from(self.last_hit)
        self.prev[:] = self.p - (away * 4.0 * S + np.array([0, 9.0 * S, 0]))
        self.hover = self.p[THX].copy()
        self.wander = wander
        self._new_target(None if wander else away)
        self.escape_until = now + seconds
        self.escape_ready = now + seconds + 1.0

    def _new_target(self, away=None) -> None:
        for _ in range(12):
            if away is None:
                t = np.array([random.uniform(-RX + 0.8, RX - 0.8), random.uniform(0.8, 2.4), random.uniform(-RZ + 0.8, RZ - 0.8)])
            else:
                t = self.p[THX] + away * random.uniform(1.3, 2.3) + np.array([0, random.uniform(0.5, 1.4), 0])
                t = np.clip(t, (-RX + 0.7, 0.8, -RZ + 0.7), (RX - 0.7, 2.4, RZ - 0.7))
            if not any(np.all((t > lo - 0.4) & (t < hi + 0.4)) for lo, hi in COLLIDERS):
                break
            away = None
        self.fly_target = t

    def step(self, now: float, pin) -> list[tuple[int, float]]:
        """One 60 Hz frame. Returns (particle, impact speed in 2D px/frame units) for hard contacts."""
        if self.frozen_at is not None:
            self.prev = self.p.copy()
            return []
        flying = now < self.escape_until and self.flying
        if flying:
            self.recover = 1.0
        elif self.grabbed is not None or now < self.stun_until or self.dead or self.wrapped:
            self.recover = 0.0
        else:
            self.recover = min(1.0, self.recover + 1 / 30)
        height = self.p[THX, 1] - STAND3
        stiff = (1 - self.melt) ** 0.5 * (1 - self.frost) * (1 - self.venom)
        strength = self.recover * (1.0 if flying else float(np.clip(1 - height / (160 * S), 0, 1)) * stiff)
        if self.grabbed is not None or (self.arena == "pool" and not flying and self.p[THX, 1] < WATER3 + 30 * S):
            strength = 0.0
        shrink = 1 - 0.35 * self.melt

        v = (self.p - self.prev) * (0.992 - 0.12 * strength)
        speed = np.linalg.norm(v, axis=1)
        v *= np.minimum(1.0, 60 * S / np.maximum(speed, 1e-9))[:, None]
        self.prev = self.p.copy()
        self.p += v
        self.p[:, 1] -= GRAV * (1 - strength)
        if self.arena == "pool":
            sub = self.p[:, 1] < WATER3
            if sub.any():
                self.p[sub, 1] += 1.35 * S
                self.p[sub] -= (self.p[sub] - self.prev[sub]) * 0.12
        if self.wind.any():
            w = np.full(N_P, 0.35)
            w[list(WING)] *= 2.5
            self.p += np.outer(w, self.wind)
            if flying:
                self.hover += self.wind * 3.0

        if flying:
            self._fly(now)
        elif strength > 0:
            self._pose(now, strength)
        else:
            self.anchor = self.p[THX, [0, 2]].copy()
        if self.dead:
            idx = list(KNEE + FOOT)
            self.p[idx] += (self.p[THX] - self.p[idx]) * 0.02
        elif now < self.flail_until or (self.grabbed is not None and random.random() < 0.5):
            idx = list(KNEE + FOOT)
            self.p[idx] += np.random.normal(0, 2.5 * S, (12, 3))

        posed = strength > 0.5
        for _ in range(6):
            if self.grabbed is not None:
                self.p[self.grabbed] = pin
            for i, at in self.stuck.items():
                self.p[i] = at
            for (a, b, stiff_k, shape), rest in zip(LINKS, LINK_LEN3):
                if shape and posed:
                    continue
                d = self.p[b] - self.p[a]
                dist = math.sqrt(float(d @ d)) or 1e-9
                corr = d * (0.5 * stiff_k * (dist - rest * shrink) / dist)
                self.p[a] += corr
                self.p[b] -= corr
            self._clamp()
        if self.grabbed is not None:
            self.p[self.grabbed] = pin
        for i, at in self.stuck.items():
            self.p[i] = at
            self.prev[i] = at
        if self.arena == "flypaper" and self.grabbed is None:
            x0, x1, z0, z1 = PAPER3
            for i in range(N_P):
                if i not in self.stuck and x0 < self.p[i, 0] < x1 and z0 < self.p[i, 2] < z1 and self.p[i, 1] <= RAD3[i] + 1.5 * S:
                    self.stuck[i] = self.p[i].copy()
        self.hurt = max(0.0, self.hurt - 1 / 20)
        return self._contacts()

    def _fly(self, now: float) -> None:
        d = self.fly_target - self.hover
        dist = float(np.linalg.norm(d))
        speed = (2.5 + 2.5 * min(self.power, 2.0)) * S
        if dist < 25 * S:
            self._new_target()
        else:
            self.hover += d / dist * min(speed, dist)
        if math.hypot(d[0], d[2]) > 8 * S:
            self.yaw_target = angle_to(d)
        self.yaw += wrap(self.yaw_target - self.yaw) * 0.15
        self.anchor = self.hover[[0, 2]].copy()
        self.action = "flying"
        tgt = to_world(REST3, self.yaw) + self.hover + (0, 4 * S * math.sin(now * 9), 0)
        f, u, _ = body_axes(self.yaw)
        for i in range(6):
            tgt[FOOT[i]] = tgt[KNEE[i]] - f * 6 * S - u * 14 * S
        dd = tgt - self.p
        self.p += dd * 0.25
        self.prev += dd * 0.2

    def _blocked(self, xz) -> bool:
        if not (-RX + 0.5 < xz[0] < RX - 0.5 and -RZ + 0.5 < xz[1] < RZ - 0.5):
            return True
        p = np.array([xz[0], 0.2, xz[1]])
        return any(np.all((p > lo - 0.35) & (p < hi + 0.35)) for lo, hi in COLLIDERS)

    def _pose(self, now: float, strength: float) -> None:
        walking, backing = now < self.walk_until, now < self.back_until
        speed = (5.0 if self.run else 2.3) * S
        v = -1.7 * S if backing else speed if walking else 0.0
        f, u, _ = body_axes(self.yaw)
        if v and self._blocked(self.anchor + f[[0, 2]] * v * 12):
            self.yaw_target = self.yaw + math.pi + random.uniform(-0.7, 0.7)    # walked into a wall: turn around
            v = 0.0
        self.yaw += float(np.clip(wrap(self.yaw_target - self.yaw), -0.09, 0.09))
        self.anchor += f[[0, 2]] * v
        self.anchor += 0.05 * (self.p[THX, [0, 2]] - self.anchor)
        self.action = "back up" if backing else ("run" if self.run else "walk") if walking else "idle"
        if v:
            self.phase += 0.12 * abs(v) / S
        s = 1 - 0.35 * self.melt
        tgt = to_world(REST3 * s, self.yaw) + (self.anchor[0], STAND3 * s + 1.5 * S * math.sin(now * 2.2), self.anchor[1])
        for leg in range(6):
            ph = self.phase + math.pi * TRIPOD[leg]
            if v:
                tgt[FOOT[leg]] += f * 14 * S * math.cos(ph) * math.copysign(1, v)
                tgt[FOOT[leg], 1] += 12 * S * max(0.0, math.sin(ph))
        d = tgt - self.p
        self.p += d * (PULL * strength)[:, None]
        self.prev += d * (PULL * strength * 0.85)[:, None]

    def _clamp(self) -> None:
        r = RAD3
        np.clip(self.p[:, 0], -RX + r, RX - r, out=self.p[:, 0])
        np.clip(self.p[:, 2], -RZ + r, RZ - r, out=self.p[:, 2])
        np.clip(self.p[:, 1], r, RY - r, out=self.p[:, 1])
        push_out_boxes(self.p, r)

    def _contacts(self) -> list[tuple[int, float]]:
        hits = []
        r = RAD3
        v = self.p - self.prev
        eps = 0.5 * S
        for i in range(N_P):
            x, y, z = self.p[i]
            if y <= r[i] + eps and v[i, 1] < 0:
                if -v[i, 1] > 9 * S:
                    hits.append((i, -v[i, 1] / S))
                self.prev[i, 1] = y + v[i, 1] * 0.35
                self.prev[i, 0] = x - v[i, 0] * 0.75
                self.prev[i, 2] = z - v[i, 2] * 0.75
            elif y >= RY - r[i] - eps and v[i, 1] > 0:
                if v[i, 1] > 9 * S:
                    hits.append((i, v[i, 1] / S))
                self.prev[i, 1] = y + v[i, 1] * 0.35
            for ax, lim in ((0, RX), (2, RZ)):
                c = self.p[i, ax]
                if (c <= -lim + r[i] + eps and v[i, ax] < 0) or (c >= lim - r[i] - eps and v[i, ax] > 0):
                    if abs(v[i, ax]) > 9 * S:
                        hits.append((i, abs(v[i, ax]) / S))
                    self.prev[i, ax] = c + v[i, ax] * 0.4
        return hits


class Player:
    def __init__(self):
        self.pos = np.array([0.0, 2.6])
        self.yaw, self.pitch = -math.pi / 2, -0.32
        self.eye_h = EYE
        self.vel = np.zeros(2)
        self.walk_phase = 0.0

    @property
    def eye(self) -> np.ndarray:
        return np.array([self.pos[0], self.eye_h + 0.012 * math.sin(self.walk_phase * 2), self.pos[1]])

    def forward(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return np.array([cp * math.cos(self.yaw), math.sin(self.pitch), cp * math.sin(self.yaw)])

    def basis(self):
        f = self.forward()
        r = np.array([-math.sin(self.yaw), 0.0, math.cos(self.yaw)])
        u = np.cross(r, f)
        return f, r, u

    def to_world(self, local) -> np.ndarray:
        """Camera space (x right, y up, -z forward) to world."""
        f, r, u = self.basis()
        return self.eye + r * local[0] + u * local[1] - f * local[2]

    def update(self, dt: float, keys, rel) -> None:
        self.yaw += rel[0] * 0.0024
        self.pitch = float(np.clip(self.pitch - rel[1] * 0.0024, -1.45, 1.45))
        fh = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        rh = np.array([-math.sin(self.yaw), math.cos(self.yaw)])
        want = fh * (keys["w"] - keys["s"]) + rh * (keys["d"] - keys["a"])
        n = float(np.linalg.norm(want))
        crouch = keys["crouch"]
        speed = 1.4 if crouch else 5.5 if keys["sprint"] else 3.0
        target = want / n * speed if n > 0 else np.zeros(2)
        self.vel += (target - self.vel) * min(1.0, dt * 12)
        self.pos += self.vel * dt
        self.pos[0] = float(np.clip(self.pos[0], -RX + 0.3, RX - 0.3))
        self.pos[1] = float(np.clip(self.pos[1], -RZ + 0.3, RZ - 0.3))
        for lo, hi in COLLIDERS:                         # slide along furniture
            q = np.array([self.pos[0], 0.5, self.pos[1]])
            if np.all((q > lo - (0.3, 0, 0.3)) & (q < hi + (0.3, 0, 0.3))):
                pen = [q[0] - (lo[0] - 0.3), (hi[0] + 0.3) - q[0], q[2] - (lo[2] - 0.3), (hi[2] + 0.3) - q[2]]
                k = int(np.argmin(pen))
                if k == 0:
                    self.pos[0] = lo[0] - 0.3
                elif k == 1:
                    self.pos[0] = hi[0] + 0.3
                elif k == 2:
                    self.pos[1] = lo[2] - 0.3
                else:
                    self.pos[1] = hi[2] + 0.3
        self.eye_h += ((CROUCH_EYE if crouch else EYE) - self.eye_h) * min(1.0, dt * 10)
        if float(np.linalg.norm(self.vel)) > 0.3:
            self.walk_phase += dt * float(np.linalg.norm(self.vel)) * 2.2


def _c(col, alpha: float | None = None):
    rgb = tuple(c / 255.0 for c in col[:3])
    a = col[3] / 255.0 if len(col) > 3 else 1.0
    return rgb + ((a if alpha is None else alpha),)


class Game3D(k2.Game):
    def __init__(self, hud: pygame.Surface, brain, view):
        self.player = Player()
        super().__init__(hud, brain, view)
        self.look = False
        self.swing_t = -9.0
        self.flick_t = -9.0
        self.throw_t = -9.0
        self.hold_dist = 1.2
        self.popups3: list = []
        self._room = self._build_room()
        self.quit_armed = False
        self.panel_mode, self.ui_mode = 0, "crisp"
        self.panel_alpha = 255
        self.view_w, self.hud_h = k2.PLAY_W, k2.H
        self.hint_extra = "V panel   F11 fullscreen   H help"

    # --- lifecycle -------------------------------------------------------------------------------------------------
    def new_fly(self) -> None:
        super().new_fly()
        pl = self.player                                  # 3 m ahead of you, side-on so you can see all of it
        start = pl.pos + np.array([math.cos(pl.yaw), math.sin(pl.yaw)]) * 3.0
        start = np.clip(start, (-RX + 0.8, -RZ + 0.8), (RX - 0.8, RZ - 0.8))
        self.fly = Fly3D(tuple(start), yaw=pl.yaw + math.pi / 2)
        self.bombs3: list = []
        self.sugars3: list = []
        self.parts: list = []                     # particles: dict(p, v, t, life, kind, size, color)
        self.bolts3: list = []
        self.shards3: list = []
        self.popups3 = []
        self.spider3: dict | None = None
        self.spider = None
        self.threat_x = self.fly.p[THX].copy()

    # --- helpers ---------------------------------------------------------------------------------------------------
    def popup(self, pos, text: str, color=(255, 245, 235), force=False) -> None:
        now = time.perf_counter()
        if self.popups3 and now - self.popups3[-1][2] < 0.3 and not force:
            return
        p = np.asarray(pos, float)
        if p.shape != (3,):
            return
        self.popups3.append([p.copy(), text, now, color])

    def puff(self, pos, n: int, spread: float = 3.0) -> None:
        pos = np.asarray(pos, float)
        if pos.shape != (3,):
            return
        for _ in range(n):
            v = np.array([random.uniform(-1, 1), random.uniform(0, 0.6), random.uniform(-1, 1)]) * spread * S
            self.parts.append(dict(p=pos.copy(), v=v, t=time.perf_counter(), life=random.uniform(0.35, 0.8),
                                   kind="dust", size=random.uniform(0.03, 0.06)))

    def aim(self):
        f, _, _ = self.player.basis()
        return self.player.eye, f

    def tool_tip(self) -> np.ndarray:
        name = TOOLS[self.tool][0]
        local = {"torch": (0.26, -0.2, 0.78), "cleaner": (0.26, -0.17, 0.66), "freeze": (0.26, -0.17, 0.66),
                 "zapper": (0.26, -0.1, 0.78), "swatter": (0.22, -0.02, 0.8)}.get(name, (0.26, -0.2, 0.62))
        return self.player.to_world(local)

    def _overlay_open(self) -> bool:
        return self.report is not None or self.big_view or self.surgery_open or self.help_open

    def _above_head(self):
        return self.fly.p[HEAD] + (0, 0.45, 0)

    # --- tools --------------------------------------------------------------------------------------------------------
    def use_tool3d(self, now: float) -> None:
        fly = self.fly
        name = TOOLS[self.tool][0]
        eye, d = self.aim()
        if fly.frozen_at is not None and fly.shattered_at is None and name in ("hand", "flick", "swatter", "zapper"):
            i, t, _ = fly.nearest_to_ray(eye, d, REACH, 0.35)
            if i is not None:
                self._shatter(now)
                return
        if name == "hand":
            i, t, _ = fly.nearest_to_ray(eye, d, GRAB_REACH, 0.18)
            if i is not None and fly.frozen_at is None:
                fly.grabbed, self.hold_dist = i, float(np.clip(t, 0.6, 2.2))
                fly.last_hit = eye.copy()
                self.hit(i, 0.25)
        elif name == "flick":
            self.flick_t = now
            i, t, _ = fly.nearest_to_ray(eye, d, 2.6, 0.4)
            if i is not None:
                center = eye + d * t
                fly.last_hit = eye.copy()
                for j in range(N_P):
                    dist = float(np.linalg.norm(fly.p[j] - center))
                    if dist < 0.45:
                        s = 1 - dist / 0.45
                        push = d * 0.7 + np.array([0, 0.6, 0])
                        fly.impulse(j, push * (10 + 16 * s) * S)
                        self.hit(j, 0.35 + 0.4 * s)
                fly.stun(now, 0.35)
                self.damage(4, "a flick")
                self.popup(center + (0, 0.25, 0), "FLICK!")
                self.sound.play("flick")
        elif name == "swatter":
            if now - self.swing_t > 0.35:
                self.swing_t = now
                self.swats.append([None, now, False])
        elif name == "bomb" and len(self.bombs3) < 3 and now - self.throw_t > 0.4:
            self.throw_t = now
            self.bombs3.append(dict(p=self.tool_tip(), v=d * 0.1 + np.array([0, 0.025, 0]), t=now))
        elif name in ("torch", "cleaner", "freeze"):
            self.torching = True
        elif name == "zapper" and now >= self.zap_ready:
            self._zap3d(now)
        elif name == "spider" and self.spider3 is None and not fly.dead:
            if d[1] < -0.05:
                pt = eye + d * (-eye[1] / d[1])
            else:
                pt = eye + np.array([d[0], 0, d[2]]) * 2.0
            pt = np.clip(pt, (-RX + 0.4, 0, -RZ + 0.4), (RX - 0.4, 0, RZ - 0.4))
            self.spider3 = dict(p=np.array([pt[0], RY - 0.05, pt[2]]), state="drop", bite_at=0.0, bites=0, anchor=pt.copy())
            self.spider = self.spider3
            self.sound.play("drop")
        elif name == "sugar" and len(self.sugars3) < 3 and now - self.throw_t > 0.3:
            self.throw_t = now
            self.sugars3.append(dict(p=self.tool_tip(), v=d * 0.07 + np.array([0, 0.02, 0]), left=1.0, landed=False))
            self.sound.play("pop")

    def _swat3d(self, now: float) -> None:
        fly = self.fly
        eye, d = self.aim()
        center = eye + d * 1.05
        self.shake_until = now + 0.15
        self.sound.play("whack")
        near = [i for i in range(N_P) if np.linalg.norm(fly.p[i] - center) < 0.8]
        if not near:
            return
        if fly.frozen_at is not None and fly.shattered_at is None:
            self._shatter(now)
            return
        fly.last_hit = eye.copy()
        dh = np.array([d[0], 0, d[2]])
        dh /= max(float(np.linalg.norm(dh)), 1e-6)
        for i in near:
            push = (fly.p[i] - center) * 0.15 + dh * 8 * S + np.array([0, -32 * S, 0])
            fly.impulse(i, push)
            self.hit(i, 1.0)
        fly.stun(now, 1.8)
        self.damage(14, "the swatter")
        self.popup(fly.p[THX] + (0, 0.45, 0), "SWAT!", (255, 230, 120))
        self.puff(np.array([fly.p[THX, 0], 0.02, fly.p[THX, 2]]), 10, 4)

    def _explode3d(self, b: dict, now: float) -> None:
        fly = self.fly
        pos = b["p"]
        self.shake_until = now + 0.4
        self.sound.play("boom")
        self.puff(pos, 24, 7)
        for _ in range(40):
            v = np.random.normal(0, 1, 3)
            v = v / np.linalg.norm(v) * random.uniform(0.02, 0.07)
            self.parts.append(dict(p=pos.copy(), v=v, t=now, life=random.uniform(0.25, 0.5), kind="fire",
                                   size=random.uniform(0.12, 0.3)))
        d = np.linalg.norm(fly.p - pos, axis=1)
        R = 330 * S
        worst = 0.0
        for i in np.flatnonzero(d < R):
            f = 48 * (1 - d[i] / R) ** 1.3
            dirv = (fly.p[i] - pos) / max(d[i], 1e-6) + np.array([0, 0.8, 0])
            fly.impulse(i, dirv * f * S)
            if f > 4:
                self.hit(i, f / 40)
                worst = max(worst, f)
        if fly.frozen_at is not None and fly.shattered_at is None and worst:
            self._shatter(now)
        elif worst:
            fly.last_hit = pos.copy()
            fly.stun(now, 2.4)
            self.damage(32 * worst / 48, "a bomb")
        self.popup(pos + (0, 0.5, 0), "KABOOM!", (255, 160, 60), force=True)

    def _zap3d(self, now: float) -> None:
        fly = self.fly
        self.zap_ready = now + 0.3
        self.sound.play("zap")
        eye, d = self.aim()
        tip = self.tool_tip()
        to = fly.p[THX] - eye
        dist = float(np.linalg.norm(to))
        i, _, _ = fly.nearest_to_ray(eye, d, REACH, 0.5)
        if (i is None and (dist > REACH or float(to @ d) / max(dist, 1e-6) < 0.93)) or fly.dissolved_at or fly.shattered_at:
            self.bolts3.append([tip, eye + d * 2.0, now])
            return
        self.bolts3.append([tip, fly.p[THX].copy(), now])
        fly.zap_until = now + 0.3
        fly.char = min(1.0, fly.char + 0.06)
        fly.last_hit = eye.copy()
        for j in range(N_P):
            fly.impulse(j, np.random.normal(0, 3 * S, 3))
        self.shake_until = now + 0.12
        if fly.dead:
            return
        for region, side in TORCH_KEYS[:-1]:
            self.brain.poke(region, side, 1.0)
        self.brain.poke("all", None, 0.0)
        fly.stun(now, 1.0)
        self.damage(16, "the zapper")
        self.popup(fly.p[THX] + (0, 0.45, 0), random.choice(("BZZZT!", "ZAP!", "KRZZT!")), (200, 235, 255))

    def _jet(self, now: float, kind: str) -> None:
        """Blowtorch flame or a spray can: a cone from the tool tip along the view."""
        fly = self.fly
        eye, d = self.aim()
        tip = self.tool_tip()
        n, spread, speed = (7, 0.12, (0.05, 0.08)) if kind == "torch" else (5, 0.18, (0.035, 0.055))
        for _ in range(n):
            jitter = np.random.normal(0, spread, 3)
            v = (d + jitter) / np.linalg.norm(d + jitter) * random.uniform(*speed)
            self.parts.append(dict(p=tip.copy(), v=v, t=now, life=random.uniform(0.22, 0.4) if kind == "torch" else random.uniform(0.4, 0.65),
                                   kind="flame" if kind == "torch" else kind, size=0.05))
        rel = fly.p - tip
        dist = np.linalg.norm(rel, axis=1)
        cosang = (rel @ d) / np.maximum(dist, 1e-6)
        rng, cone = (1.9, 0.85) if kind == "torch" else (2.0, 0.8)
        inside = np.flatnonzero((dist < rng) & (cosang > cone))
        if kind == "torch":
            if len(inside):
                fly.burn_until = now + 0.9
                for i in inside:
                    fly.impulse(i, d * 0.5 * S + np.array([0, 0.2 * S, 0]))
            if now >= fly.burn_until:
                return
            fly.char = min(1.0, fly.char + 0.004)
            fly.hurt = max(fly.hurt, 0.5)
            fly.last_hit = eye.copy()
            if fly.dead:
                return
            for region, side in TORCH_KEYS:
                self.brain.poke(region, side, 1.0)
            self.damage(0.35, "the blowtorch")
            words, col = ("SIZZLE!", "TSSSS!", "HOT HOT!"), (255, 150, 60)
        else:
            if fly.dissolved_at is not None or fly.frozen_at is not None or not len(inside):
                return
            fly.last_hit = eye.copy()
            if kind == "cleaner":
                fly.soak = min(1.0, fly.soak + 0.04)
            else:
                fly.frost = min(0.9 if self.immortal else 1.0, fly.frost + 0.006)
            if fly.dead:
                return
            if kind == "cleaner":
                self.brain.poke("smell", None, 1.0)
                self.brain.poke("taste", None, 0.8)
                self.damage(0.12, "brake cleaner")
                words, col = ("FSSSSH!", "MELTING!", "IT BURNS!"), (170, 230, 255)
            else:
                self.brain.poke("cold", None, 1.0)
                for region, side in TORCH_KEYS[:-1]:
                    self.brain.poke(region, side, 0.35)
                self.damage(0.1, "freezing")
                words, col = ("SO COLD!", "BRRRR!", "ICING!"), (190, 230, 255)
        if int(now * 2) != int((now - 1 / 60) * 2):
            self.hits += 1
        if random.random() < 0.02:
            self.popup(fly.p[HEAD] + (0, 0.4, 0), random.choice(words), col)

    def _shatter(self, now: float) -> None:
        fly = self.fly
        fly.shattered_at = now
        self.shake_until = now + 0.25
        self.sound.play("shatter")
        for i in range(N_P):
            for _ in range(3):
                self.shards3.append(dict(p=fly.p[i].copy(), v=np.random.normal(0, 0.02, 3) + (0, 0.03, 0),
                                         rot=np.random.uniform(0, 6.28, 3), size=random.uniform(0.02, 0.05),
                                         fly=random.random() < 0.35))
        self.popup(fly.p[THX] + (0, 0.5, 0), "SHATTERED!", (200, 235, 255), force=True)

    # --- the fly's senses ------------------------------------------------------------------------------------------------
    def _threats(self, now: float, mouse=None) -> list:
        out = []
        if not self._overlay_open():
            eye = self.player.eye
            out.append(("player", eye - (0, 0.35, 0), 0.28))
            name = TOOLS[self.tool][0]
            if name not in ("hand", "sugar"):
                out.append(("tool", self.tool_tip(), TOOL_SIZE.get(name, 0.08)))
            ph = now - self.swing_t
            if ph < 0.14:
                _, d = self.aim()
                a = self.player.to_world((0.28, 0.35, 0.35))
                b = self.player.eye + d * 1.05
                out.append(("swing", a + (b - a) * (ph / 0.14), 0.2))
        if self.spider3 is not None and self.spider3["state"] in ("drop", "hunt"):
            out.append(("spider", self.spider3["p"].copy(), 0.1))
        for b in self.bombs3:
            out.append((("bomb", id(b)), b["p"].copy(), 0.08))
        return out

    def _vision(self, now: float, mouse=None) -> None:
        fly = self.fly
        if fly.dead or fly.frozen_at is not None:
            self.loom = 0.0
            return
        head = fly.p[HEAD]
        best, best_pos, seen = 0.0, None, {}
        for key, pos, r in self._threats(now):
            dist = max(float(np.linalg.norm(pos - head)), r + 0.02)
            theta = 2 * math.atan(r / dist)
            prev = self.loom_prev.get(key)
            seen[key] = theta
            if prev is not None and (theta - prev) * 60.0 > best:
                best, best_pos = (theta - prev) * 60.0, pos
        self.loom_prev = seen
        self.loom += (best - self.loom) * 0.5
        strength = float(np.clip((best - k2.LOOM_MIN) / k2.LOOM_FULL, 0, 1))
        if strength > 0 and best_pos is not None:
            self.brain.poke("loom", None, strength, recruit=0.6 * strength)
            self.threat_x = np.asarray(best_pos, float).copy()

    def _scents(self, now: float, mouse=None) -> None:
        fly = self.fly
        self.scent_now, self.sugar_scent = None, False
        if fly.dead:
            return
        head = fly.p[HEAD]
        if not self._overlay_open() and np.linalg.norm(self.tool_tip() - head) < 2.5:
            self.scent_now = TOOLS[self.tool][0]
            self.brain.poke("scent", self.scent_now, 0.3)
        if any(np.linalg.norm((s["p"] - head)[[0, 2]]) < 2.4 for s in self.sugars3):
            self.sugar_scent = True
            self.brain.poke("scent", "sugar", 0.3)

    def _memory_behavior(self, now: float, free: bool, can_fly: bool) -> None:
        fly = self.fly
        if not self.scent_now or not free or now < self.avoid_ready or fly.frozen_at is not None or now < fly.stun_until:
            return
        eye = self.player.eye
        if self.scent_now != "sugar" and self.fear_now > k2.FEAR_ACT:
            self.avoid_ready = now + 2.5
            fly.last_hit = eye.copy()
            if can_fly and self.fear_now > 0.6:
                fly.escape(now)
            else:
                fly.yaw_target = angle_to(fly.away_from(eye))
                fly.walk_until, fly.back_until, fly.run = now + 1.2, 0.0, True
            self.note(f"AVOID    remembers the {self.scent_now} ({self.fear_now:.2f})")
            self.popup(fly.p[HEAD] + (0, 0.4, 0), "NOPE!", (255, 220, 120))
        elif self.scent_now == "sugar" and self.like_now > k2.LIKE_ACT and now >= fly.walk_until:
            self.avoid_ready = now + 1.5
            fly.yaw_target = angle_to(-fly.away_from(eye))
            fly.walk_until, fly.run = now + 1.0, False
            self.note(f"APPROACH remembers sugar ({self.like_now:.2f})")

    # --- arenas and ongoing effects -------------------------------------------------------------------------------------------
    def _environment(self, now: float, mouse=None) -> None:
        fly, br = self.fly, self.brain
        arena = k2.ARENAS[self.arena_i]
        fly.arena, fly.wind = arena, np.zeros(3)
        if arena != "flypaper":
            fly.stuck.clear()
        if arena == "fan":
            gust = 0.75 + 0.25 * math.sin(now * 1.3) + 0.15 * math.sin(now * 4.1)
            dx = fly.p[THX, 0] - FAN3[0]
            lateral = math.exp(-((fly.p[THX, 2] - FAN3[2]) ** 2) / (2 * 1.9 ** 2))
            w = 0.55 * gust * float(np.clip(1.15 - dx / 7.0, 0.25, 1.0)) * lateral
            fly.wind = np.array([w * S, 0, 0])
            if not fly.dead and self.frame % 3 == 0:
                br.poke("wind", None, min(1.0, w * 1.4))
            if random.random() < 0.8:
                self.parts.append(dict(p=np.array([FAN3[0] + 0.5, random.uniform(0.4, 1.6), FAN3[2] + random.uniform(-1.2, 1.2)]),
                                       v=np.array([random.uniform(0.09, 0.15), 0, 0]), t=now, life=1.0, kind="streak", size=0.025))
        elif arena == "flypaper" and fly.stuck:
            kicking = now < fly.flail_until
            for i in list(fly.stuck):
                pull = fly.grabbed is not None and np.linalg.norm(self._hold_point() - fly.stuck[i]) > 0.55
                if random.random() < 0.0012 + (0.012 if kicking else 0) + (0.06 if pull else 0):
                    del fly.stuck[i]
            if not fly.dead:
                if self.frame % 6 == 0:
                    br.poke("legs", "L", 0.4)
                    br.poke("legs", "R", 0.4)
                    br.poke("body", None, 0.2)
                if len(fly.stuck) >= 3:
                    self.damage(0.012, "the flypaper")
        elif arena == "pool":
            sub = fly.p[:, 1] < WATER3
            if sub.any():
                if fly.wet <= 0 and float(np.max((fly.prev - fly.p)[sub, 1])) > 4 * S:
                    self.sound.play("splash")
                    self.puff(np.array([fly.p[THX, 0], WATER3, fly.p[THX, 2]]), 10, 3)
                fly.wet = 3.0
                if not fly.dead:
                    if self.frame % 4 == 0:
                        br.poke("humid", None, 0.9)
                        br.poke("body", None, 0.25)
                    if fly.p[HEAD, 1] < WATER3 - 6 * S:
                        self.damage(0.05, "drowning")
        elif arena == "lamp":
            dist = float(np.linalg.norm(fly.p[HEAD] - LAMP3))
            if not fly.dead:
                light = float(np.clip(1.2 - dist / 3.0, 0.15, 1.0))
                if self.frame % 2 == 0:
                    br.poke("light", None, light, recruit=0.25 * light)
                if dist < 0.36:
                    br.poke("heat", None, 0.8)
                    self.damage(0.08, "the hot lamp")
                    away = (fly.p[HEAD] - LAMP3) / max(dist, 1e-6)
                    for i in (HEAD, THX, ABD):
                        fly.impulse(i, away * 1.5 * S)
            if now < fly.escape_until and np.linalg.norm(fly.fly_target - LAMP3) > 0.8:
                fly.fly_target = LAMP3 + np.array([random.uniform(-0.6, 0.6), -random.uniform(0.3, 0.8), random.uniform(-0.6, 0.6)])
        if arena != "pool" or not (fly.p[:, 1] < WATER3).any():
            fly.wet = max(0.0, fly.wet - 1 / 60)

    def _kick(self, now: float) -> None:
        """Your legs are a 0.28 m cylinder: walking into the fly shoves it, and walking into it fast is a kick."""
        fly, pl = self.fly, self.player
        if fly.frozen_at is not None or fly.dissolved_at is not None or fly.shattered_at is not None:
            return
        rel = fly.p[:, [0, 2]] - pl.pos
        dist = np.linalg.norm(rel, axis=1)
        inside = np.flatnonzero((dist < 0.28 + RAD3) & (fly.p[:, 1] < 0.9))
        if not len(inside):
            return
        speed = float(np.linalg.norm(pl.vel))
        for i in inside:
            n = rel[i] / max(dist[i], 1e-6)
            fly.p[i, [0, 2]] = pl.pos + n * (0.28 + RAD3[i])
        if speed > 1.2 and now - getattr(self, "kick_t", 0) > 0.4:
            self.kick_t = now
            push = np.append(pl.vel, 0)[[0, 2, 1]] / 60 * 1.6 + np.array([0, 0.012 + 0.006 * speed, 0])
            for i in range(N_P):
                fly.impulse(i, push)
            for i in inside:
                self.hit(i, min(1.0, speed / 5))
            fly.last_hit = pl.eye.copy()
            if speed > 2.5:
                fly.stun(now, 0.6)
                self.damage(3 + 2 * speed, "a kick")
                self.popup(fly.p[THX] + (0, 0.4, 0), random.choice(("KICK!", "BOOT!", "PUNT!")), (255, 235, 150))
                self.sound.play("whack", 0.6)

    def _hold_point(self) -> np.ndarray:
        eye, d = self.aim()
        return eye + d * self.hold_dist

    def _effects(self, now: float) -> None:
        fly = self.fly
        if self.immortal:
            fly.melt = min(fly.melt, 0.85)
            if fly.soak < 0.05:
                fly.melt = max(0.0, fly.melt - 0.002)
            fly.venom = max(0.0, fly.venom - 0.002)
        if fly.dissolved_at is None and fly.soak > 0.02:
            fly.melt = min(0.85 if self.immortal else 1.0, fly.melt + 0.0045 * fly.soak)
            fly.soak *= 0.985 if self.immortal else 0.997
            if not fly.dead:
                self.damage(0.25 * fly.soak, "brake cleaner")
                for region, side in TORCH_KEYS[:-1]:
                    self.brain.poke(region, side, 0.5 * fly.soak)
        if fly.melt >= 1.0 and fly.dissolved_at is None:
            fly.dissolved_at = now
            self.puff(np.array([fly.p[THX, 0], 0.05, fly.p[THX, 2]]), 14, 3)
            self.popup(fly.p[THX] + (0, 0.4, 0), "DISSOLVED", (170, 230, 255), force=True)
            self.sound.play("squish")
            if not fly.dead:
                self.damage(MAX_HEALTH, "brake cleaner")
        if fly.frozen_at is None and fly.frost > 0:
            if not (self.torching and TOOLS[self.tool][0] == "freeze"):
                fly.frost = max(0.0, fly.frost - 0.0015)
            if fly.frost >= 1.0:
                fly.frozen_at = now
                self.popup(fly.p[THX] + (0, 0.45, 0), "FROZEN SOLID", (190, 230, 255), force=True)
                if not fly.dead:
                    self.damage(MAX_HEALTH, "freezing")
        if not fly.dead:
            self.brain.sedation = max(0.25 * fly.melt ** 2.5, 0.2 * fly.frost ** 2, 0.3 * fly.venom ** 1.5)
        self._spider3d(now)
        self._sugar3d(now)

    def _spider3d(self, now: float) -> None:
        sp, fly = self.spider3, self.fly
        if sp is None:
            return
        if sp["state"] == "drop":
            sp["p"][1] -= 0.04
            if sp["p"][1] <= 0.12:
                sp["p"][1], sp["state"] = 0.12, "hunt"
            return
        if sp["state"] == "hunt":
            if fly.dead or fly.dissolved_at is not None or fly.shattered_at is not None:
                sp["state"] = "leave"
                return
            d = fly.p[THX] + (0, 0.1, 0) - sp["p"]
            dist = float(np.linalg.norm(d))
            if dist > 0.2:
                sp["p"] += d / dist * min(3.4 * S, dist)
                sp["p"][1] = max(0.12, sp["p"][1])
                sp["anchor"] = sp["p"].copy()
            elif now - sp["bite_at"] > 0.7:
                sp["bite_at"] = now
                sp["bites"] += 1
                fly.venom = min(1.0, fly.venom + 0.3)
                fly.last_hit = sp["p"].copy()
                fly.hurt = 1.0
                self.brain.poke("body", None, 0.9)
                self.brain.poke("legs", "L", 0.6)
                self.brain.poke("legs", "R", 0.6)
                self.damage(16, "a spider")
                self.sound.play("chomp")
                self.popup(sp["p"] + (0, 0.3, 0), random.choice(("CHOMP!", "BITE!", "SLURP!")), (230, 120, 120))
                if sp["bites"] >= 2 and not fly.wrapped:
                    fly.wrapped = True
                    self.note("WRAPPED  in spider silk")
                if self.immortal and sp["bites"] >= 5:
                    fly.wrapped, fly.grabbed = False, None
                    sp["state"] = "leave"
                    self.popup(fly.p[HEAD] + (0, 0.4, 0), "BROKE FREE!", (255, 225, 120), force=True)
                    self.note("BROKE FREE of the silk")
                    return
            if fly.wrapped:
                fly.grabbed = THX
        elif sp["state"] in ("leave", "carry"):
            if fly.wrapped and fly.dead and fly.frozen_at is None:
                sp["state"] = "carry"
            sp["p"][1] += 2.2 * S * 2
            if sp["state"] == "carry":
                fly.grabbed = THX
            if sp["p"][1] > RY + 0.4:
                self.spider3 = self.spider = None
                if fly.wrapped:
                    fly.grabbed = None

    def _sugar3d(self, now: float) -> None:
        fly = self.fly
        for s in self.sugars3:
            if not s["landed"]:
                s["v"][1] -= GRAV
                s["p"] += s["v"]
                for ax, lim in ((0, RX - 0.05), (2, RZ - 0.05)):
                    if abs(s["p"][ax]) > lim:
                        s["p"][ax] = math.copysign(lim, s["p"][ax])
                        s["v"][ax] *= -0.4
                if s["p"][1] <= 0.035:
                    s["p"][1], s["landed"] = 0.035, True
        if not self.sugars3 or fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None:
            return
        s = min(self.sugars3, key=lambda s: float(np.linalg.norm((s["p"] - fly.p[THX])[[0, 2]])))
        dxz = (s["p"] - fly.p[HEAD])[[0, 2]]
        on_floor = fly.p[THX, 1] < STAND3 + 0.2 and now >= fly.escape_until
        if float(np.linalg.norm(dxz)) > 0.2:
            if on_floor and now >= fly.stun_until and s["landed"]:
                fly.yaw_target = math.atan2(dxz[1], dxz[0])
                fly.walk_until, fly.run = now + 0.2, False
            return
        if not on_floor or not s["landed"]:
            return
        if now >= fly.eating_until:
            self.popup(fly.p[HEAD] + (0, 0.4, 0), random.choice(("YUM!", "SWEET!", "NOM NOM")), (255, 160, 190))
            self.sound.play("yum")
            self.note("EATING   sugar: taste + PAM reward")
        fly.eating_until = now + 0.4
        s["left"] -= 1 / 240
        self.brain.poke("taste", None, 0.5)
        self.brain.poke("reward", None, 0.4)
        fly.health = min(MAX_HEALTH, fly.health + 0.15)
        if s["left"] <= 0:
            self.sugars3.remove(s)

    def _die(self, now: float) -> None:
        fly = self.fly
        fly.dead_at = now
        fly.grabbed = None
        self.killed_by = self.damage_src or "being kicked"
        self.kills += 1
        self.brain.kill()
        self.sound.play("death")
        self.note("DIED     brain drive cut, activity fading")
        self.popup(fly.p[HEAD] + (0, 0.5, 0), "K.O.!", (255, 90, 80), force=True)
        self.shake_until = now + 0.3

    def _fly_state(self, now: float) -> str:
        f = self.fly
        for cond, word in ((f.shattered_at is not None, "shattered"), (f.dissolved_at is not None, "dissolved"),
                           (f.frozen_at is not None, "frozen solid"), (f.dead, "dead"), (bool(f.stuck), "stuck on flypaper")):
            if cond:
                return word
        if f.arena == "pool" and (f.p[:, 1] < WATER3).any() and now >= f.escape_until:
            return "swimming"
        if f.wrapped:
            return "wrapped in silk"
        if f.grabbed is not None:
            return "grabbed"
        if now < f.escape_until:
            return "flying"
        if now < f.eating_until:
            return "eating sugar"
        if now < f.stun_until:
            return "stunned"
        return f.action

    # --- per frame ----------------------------------------------------------------------------------------------------------
    def update3d(self, now: float, dt: float, keys, rel) -> None:
        fly, br = self.fly, self.brain
        self.frame += 1
        if self.look:
            self.player.update(dt, keys, rel)
        self._environment(now)
        self._kick(now)
        pin = self._hold_point()
        if fly.wrapped and self.spider3 is not None:
            pin = self.spider3["p"] - (0, 0.16, 0)
        for i, sp in fly.step(now, pin):
            s = float(np.clip((sp - 9) / 35, 0.05, 1))
            self.hit(i, s)
            if sp > 18:
                self.damage(min(8.0, (sp - 18) * 0.35), "the floor" if fly.p[i, 1] < 0.2 else "the wall")
            if sp > 14 and fly.p[i, 1] < 0.2:
                self.puff(np.array([fly.p[i, 0], 0.02, fly.p[i, 2]]), 3)
            if sp > 24 and not fly.dead:
                self.sound.play("bonk", 0.4 + 0.6 * s)
                fly.stun(now, 0.8 * s + 0.3)
                if random.random() < 0.5:
                    self.popup(fly.p[i] + (0, 0.3, 0), random.choice(k2.OUCH))
        for sw in self.swats:
            if not sw[2] and now - sw[1] > 0.12:
                sw[2] = True
                self._swat3d(now)
        self.swats = [s for s in self.swats if now - s[1] < 0.55]
        for b in list(self.bombs3):
            b["v"][1] -= GRAV
            b["p"] += b["v"]
            if b["p"][1] < 0.08:
                b["p"][1] = 0.08
                b["v"][1] *= -0.35
                b["v"][[0, 2]] *= 0.7
            for ax, lim in ((0, RX - 0.08), (2, RZ - 0.08)):
                if abs(b["p"][ax]) > lim:
                    b["p"][ax] = math.copysign(lim, b["p"][ax])
                    b["v"][ax] *= -0.5
            push_out_boxes(b["p"][None, :], 0.08)
            if now - b["t"] > 1.8:
                self.bombs3.remove(b)
                self._explode3d(b, now)
        if self.torching and self.report is None and TOOLS[self.tool][0] in ("torch", "cleaner", "freeze"):
            self._jet(now, TOOLS[self.tool][0])
        self._effects(now)
        for p in self.parts:
            p["p"] = p["p"] + p["v"]
            if p["kind"] == "flame":
                p["v"] = p["v"] * 0.95 + (0, 0.0015, 0)
            elif p["kind"] in ("cleaner", "freeze"):
                p["v"] = p["v"] * 0.93 + (0, -0.0003, 0)
            elif p["kind"] == "dust":
                p["v"] = p["v"] * 0.96 + (0, 0.0004, 0)
            elif p["kind"] == "fire":
                p["v"] = p["v"] * 0.9 + (0, 0.001, 0)
        self.parts = [p for p in self.parts if now - p["t"] < p["life"]]
        if len(self.parts) > 1500:
            self.parts = self.parts[-1500:]
        for sh in self.shards3:
            sh["v"][1] -= GRAV
            sh["p"] += sh["v"]
            if sh["p"][1] < sh["size"]:
                sh["p"][1] = sh["size"]
                sh["v"] *= (0.6, -0.3, 0.6)
            sh["rot"] += sh["v"] * 8
        self.bolts3 = [b for b in self.bolts3 if now - b[2] < 0.25]
        self.popups3 = [pu for pu in self.popups3 if now - pu[2] < 0.9]

        parts = br.pain_parts(br.fast, br.base)[0]
        parts[3] = max(parts[3], self.pain_parts[3] * 0.96)
        self.pain_parts += (parts - self.pain_parts) * 0.15
        self.pain += (float(br.pain_index(parts)) - self.pain) * 0.15
        self.reward += (float(np.clip((br.level("reward") - 1.0) / 0.6, 0, 1)) * 100 - self.reward) * 0.1
        if not fly.dead and self.pain > 25 and self.frame % 3 == 0:
            br.poke("punish", None, self.pain / 100)
        self._vision(now)
        self._scents(now)
        self._learn(now)
        self._sound_update(now)
        if not fly.dead:
            self.pain_peak = max(self.pain_peak, self.pain)
            if self.pain >= 99:
                self.pain_max_s += 1 / 60
        self.pain_trace.append(self.pain)
        self.pain_trace = self.pain_trace[-720:]

        if not fly.dead:
            for (region, side), s in self.pending_hits.items():
                br.poke(region, side, s)
                self.hits += 1
            if self.pending_damage:
                floor = 1.0 if self.immortal else 0.0
                fly.health = max(floor, fly.health - self.pending_damage)
                self.last_damage = now
                if fly.health <= 0:
                    self._die(now)
            elif self.immortal and now - self.last_damage > 1.5:
                fly.health = min(MAX_HEALTH, fly.health + 0.1)
        self.pending_hits.clear()
        self.pending_damage = 0.0
        if fly.dead:
            if self.report is None and now - fly.dead_at > k2.AUTOPSY_DELAY:
                self.report = self._autopsy(now)
                self.death_frames = list(self.frames)
            return

        # reactions read from the descending neurons
        can_fly = fly.grabbed is None and not fly.wrapped and fly.frost < 0.5 and fly.melt < 0.3 and fly.venom < 0.5
        can_fly = can_fly and fly.wet <= 0 and len(fly.stuck) < 2
        free = fly.grabbed is None and now >= fly.escape_until
        lv = {n: br.level(n) for n in ("jump", "run", "kick", "walk", "back", "turn_l", "turn_r", "fly", "escape")}
        fly.power = lv["fly"]
        if lv["escape"] > THRESH["escape"] and now >= fly.escape_ready and fly.grabbed is None and can_fly:
            fly.stun_until = 0.0
            fly.last_hit = np.asarray(self.threat_x, float).copy()
            fly.escape(now)
            self.note(f"DODGE    giant fiber DNp01 x{lv['escape']:.1f}")
            self.popup(fly.p[HEAD] + (0, 0.4, 0), "DODGE!", (170, 255, 200))
            self.sound.play("dodge")
        elif lv["jump"] > THRESH["jump"] and now >= fly.escape_ready and free and can_fly:
            fly.stun_until = 0.0
            fly.escape(now)
            self.note(f"FLY AWAY head-touch DNs x{lv['jump']:.1f}")
            self.popup(fly.p[HEAD] + (0, 0.4, 0), "YIKES!", (160, 230, 255))
        elif (lv["fly"] > THRESH["fly"] and now >= fly.escape_ready and free and can_fly and now >= fly.stun_until
              and now >= fly.eating_until):
            fly.escape(now, seconds=random.uniform(2.5, 4.0), wander=True)
            self.note(f"TAKE OFF DNg02 x{lv['fly']:.2f}")
        if lv["run"] > THRESH["run"] and now >= fly.walk_until and free:
            fly.stun_until = min(fly.stun_until, now + 0.2)
            fly.yaw_target = angle_to(fly.away_from(fly.last_hit))
            fly.walk_until, fly.back_until, fly.run = now + 1.1, 0.0, True
            self.note(f"RUN      body-touch DNs x{lv['run']:.1f}")
        if lv["kick"] > THRESH["kick"] and now >= fly.flail_until:
            fly.flail_until = now + 0.6
            self.note(f"KICK     leg-touch DNs x{lv['kick']:.1f}")
        if lv["back"] > THRESH["back"] and now >= fly.back_until and now >= fly.walk_until:
            fly.back_until = now + 0.8
            self.note(f"BACK UP  MDN x{lv['back']:.1f}")
        elif lv["walk"] > THRESH["walk"] and now >= fly.walk_until and now >= fly.back_until:
            fly.walk_until, fly.run = now + 1.2, False
            self.note(f"WALK     DNp09 x{lv['walk']:.1f}")
        self._memory_behavior(now, free, can_fly)
        lamp_idle = free and now >= fly.escape_until and now >= fly.stun_until and now >= fly.walk_until
        if k2.ARENAS[self.arena_i] == "lamp" and lamp_idle and now >= self.photo_ready:
            self.photo_ready = now + random.uniform(3.0, 6.0)
            fly.yaw_target = angle_to(LAMP3 - fly.p[THX])
            if can_fly and random.random() < 0.6:
                fly.escape(now, seconds=random.uniform(3.0, 5.0), wander=True)
                fly.fly_target = LAMP3 - (0, 0.5, 0)
                self.note("TO LIGHT flies to the lamp")
            else:
                fly.walk_until, fly.run = now + 1.5, False
        turn = lv["turn_r"] - lv["turn_l"]
        if abs(turn) > THRESH["turn"] and now >= fly.turn_ready and free and now >= fly.walk_until:
            fly.turn_ready = now + 1.5
            fly.yaw_target = fly.yaw + math.copysign(random.uniform(0.9, 1.6), turn)
            self.note(f"TURN {'R' if turn > 0 else 'L'}   DNa01/02 R-L {turn:+.1f}")

    # --- world drawing --------------------------------------------------------------------------------------------------------
    def _build_room(self) -> list:
        R = []

        def box(center, size, color, pattern=P_NONE, glow=0.0):
            R.append(("cube", trs(center, None, size), color, pattern, glow))

        box((0, -0.05, 0), (2 * RX, 0.1, 2 * RZ), (0.66, 0.46, 0.28), P_WOOD)
        box((0, RY + 0.05, 0), (2 * RX, 0.1, 2 * RZ), (0.93, 0.92, 0.9), P_CEIL)
        wall = (0.8, 0.73, 0.6)
        box((0, RY / 2, -RZ - 0.05), (2 * RX, RY, 0.1), wall, P_WALLPAPER)
        box((0, RY / 2, RZ + 0.05), (2 * RX, RY, 0.1), wall, P_WALLPAPER)
        box((-RX - 0.05, RY / 2, 0), (0.1, RY, 2 * RZ), wall, P_WALLPAPER)
        box((RX + 0.05, RY / 2, 0), (0.1, RY, 2 * RZ), wall, P_WALLPAPER)
        trim = (0.94, 0.92, 0.88)
        for y, h in ((0.06, 0.12), (0.95, 0.04), (RY - 0.04, 0.08)):
            box((0, y, -RZ + 0.015), (2 * RX, h, 0.03), trim)
            box((0, y, RZ - 0.015), (2 * RX, h, 0.03), trim)
            box((-RX + 0.015, y, 0), (0.03, h, 2 * RZ), trim)
            box((RX - 0.015, y, 0), (0.03, h, 2 * RZ), trim)
        # window on the far wall
        box((1.4, 1.75, -RZ + 0.012), (1.9, 1.25, 0.02), (1, 1, 1), 8)
        for dx, dy, sx, sy in ((0, 0.66, 2.02, 0.08), (0, -0.66, 2.02, 0.08), (-0.99, 0, 0.08, 1.4), (0.99, 0, 0.08, 1.4),
                               (0, 0, 0.05, 1.3), (0, 0, 1.95, 0.05)):
            box((1.4 + dx, 1.75 + dy, -RZ + 0.04), (sx, sy, 0.06), trim)
        box((1.4, 1.08, -RZ + 0.1), (2.1, 0.05, 0.2), trim)
        # couch
        blue, cushion = (0.33, 0.4, 0.55), (0.42, 0.5, 0.66)
        box((-2.3, 0.22, -RZ + 0.55), (2.3, 0.36, 0.9), blue)
        box((-2.3, 0.72, -RZ + 0.16), (2.3, 0.75, 0.26), blue)
        box((-3.39, 0.45, -RZ + 0.55), (0.22, 0.52, 0.95), blue)
        box((-1.21, 0.45, -RZ + 0.55), (0.22, 0.52, 0.95), blue)
        box((-2.83, 0.48, -RZ + 0.6), (1.0, 0.16, 0.7), cushion)
        box((-1.77, 0.48, -RZ + 0.6), (1.0, 0.16, 0.7), cushion)
        # bookshelf
        dark = (0.33, 0.22, 0.14)
        box((RX - 0.22, 1.02, -1.0), (0.42, 2.04, 1.8), dark)
        for k in range(4):
            box((RX - 0.24, 0.3 + k * 0.48, -1.0), (0.36, 0.34, 1.62), (1, 1, 1), P_BOOKS)
            box((RX - 0.22, 0.12 + k * 0.48, -1.0), (0.4, 0.03, 1.76), dark)
        # plant
        R.append(("cylinder", trs((RX - 0.6, 0, RZ - 0.6), None, (0.25, 0.45, 0.25)), (0.7, 0.35, 0.22), P_NONE, 0.0))
        for k in range(7):
            a = k * 0.9
            R.append(("sphere", trs((RX - 0.6 + 0.2 * math.cos(a), 0.75 + 0.12 * (k % 3), RZ - 0.6 + 0.2 * math.sin(a)), None,
                                    (0.24, 0.3, 0.24)), (0.24, 0.52, 0.26), P_NONE, 0.0))
        # door behind you, a picture, the rug and the ceiling light
        box((-2.2, 1.05, RZ - 0.03), (0.95, 2.1, 0.05), (0.52, 0.34, 0.2))
        R.append(("sphere", trs((-1.85, 1.0, RZ - 0.08), None, (0.04, 0.04, 0.04)), (0.85, 0.72, 0.35), P_NONE, 0.3))
        box((-RX + 0.03, 1.75, 1.2), (0.04, 0.8, 1.1), (0.25, 0.18, 0.12))
        box((-RX + 0.055, 1.75, 1.2), (0.02, 0.66, 0.96), (0.9, 0.62, 0.3), P_RUG, 0.1)
        R.append(("cylinder", trs((-0.4, 0.0, -0.9), None, (1.35, 0.012, 1.0)), (0.9, 0.52, 0.36), P_RUG, 0.0))
        R.append(("cylinder", trs((0, RY - 0.06, 0), None, (0.3, 0.06, 0.3)), (1.0, 0.97, 0.9), P_NONE, 1.3))
        return R

    def draw_world(self, rd: Renderer, now: float) -> None:
        for mesh, model, color, pattern, glow in self._room:
            rd.add(mesh, model, color, pattern, glow)
        arena = k2.ARENAS[self.arena_i]
        if arena == "fan":
            self._draw_fan(rd, now)
        elif arena == "flypaper":
            x0, x1, z0, z1 = PAPER3
            rd.add("cube", trs(((x0 + x1) / 2, 0.018, (z0 + z1) / 2), None, (x1 - x0, 0.01, z1 - z0)), (0.88, 0.72, 0.24), P_PAPER)
        elif arena == "pool":
            rd.add("cube", trs((0, WATER3, 0), None, (2 * RX, 0.01, 2 * RZ)), (0.18, 0.42, 0.62, 0.55), P_WATER)
            rd.add("cube", trs((0, WATER3 / 2, 0), None, (2 * RX - 0.02, WATER3, 2 * RZ - 0.02)), (0.1, 0.3, 0.45, 0.25), P_NONE)
        elif arena == "lamp":
            top = np.array([LAMP3[0], RY, LAMP3[2]])
            rd.add("cylinder", segment(LAMP3 + (0, 0.2, 0), top, 0.008), (0.15, 0.15, 0.15))
            rd.add("cylinder", trs(LAMP3 + (0, 0.02, 0), None, (0.26, 0.2, 0.26)), (0.25, 0.32, 0.28))
            rd.add("sphere", trs(LAMP3, None, (0.11, 0.13, 0.11)), (1.0, 0.95, 0.8), P_NONE, 3.0)
            for k in range(3):
                rd.particle(LAMP3, 0.35 + 0.25 * k + 0.03 * math.sin(now * 3 + k), (1.0, 0.85, 0.5, 0.12), additive=True)
        self._draw_fly(rd, now)
        self._draw_extras(rd, now)

    def _draw_fan(self, rd: Renderer, now: float) -> None:
        base = FAN3
        hub = base + (0, 1.0, 0)
        grey = (0.72, 0.74, 0.78)
        rd.add("cylinder", trs(base, None, (0.3, 0.05, 0.3)), grey)
        rd.add("cylinder", segment(base, hub - (0.1, 0, 0), 0.03), grey)
        rd.add("sphere", trs(hub - (0.12, 0, 0), None, (0.16, 0.14, 0.14)), grey)
        rim = rot_z(math.pi / 2)
        rd.add("torus", trs(hub + (0.1, 0, 0), rim, (0.48, 0.48, 0.48)), (0.85, 0.87, 0.9))
        for k in range(3):
            a = now * 22 + k * 2.094
            offset = np.array([0.06, math.cos(a), math.sin(a)]) * (1, 0.22, 0.22)
            blade = rot_x(a) @ np.eye(3)
            rd.add("sphere", trs(hub + offset, blade, (0.03, 0.22, 0.1)), (0.6, 0.8, 0.95, 0.85))
        rd.add("sphere", trs(hub + (0.08, 0, 0), None, (0.05, 0.05, 0.05)), (0.3, 0.3, 0.35))

    def _draw_fly(self, rd: Renderer, now: float) -> None:
        fly = self.fly
        p = fly.p
        hurt, dead = fly.hurt, fly.dead
        if fly.dissolved_at is not None:
            e = min(1.0, (now - fly.dissolved_at) / 1.5)
            x, z = p[THX, 0], p[THX, 2]
            rd.add("cylinder", trs((x, 0.006, z), None, ((70 + 60 * e) * S, 0.012, (60 + 50 * e) * S)), (0.43, 0.41, 0.24, 0.85))
            rd.add("sphere", trs((x + 0.15, 0.03, z), rot_y(0.4), (0.16, 0.01, 0.08)), (0.8, 0.85, 0.93, 0.4))
            for dx in (-0.05, 0.04):
                rd.add("sphere", trs((x + dx, 0.025, z + 0.02), None, (0.018, 0.018, 0.018)), (0.7, 0.1, 0.1))
            for k in range(3):
                ph = (now * 0.8 + k / 3) % 1.0
                rd.particle((x + (k - 1) * 0.25, 0.03 + 0.08 * ph, z), 0.02 + 0.02 * ph, (0.85, 0.85, 0.7, 0.6 * (1 - ph)))
            return
        if fly.shattered_at is not None:
            return
        zapped = now < fly.zap_until and int(now * 40) % 2 == 0
        k_ = 1 - 0.4 * fly.melt

        def col(c):
            if zapped:
                return (0.8, 0.92, 1.0, 1.0)
            return _c(k2.shade(c, hurt, dead, fly.char, fly.melt, fly.frost))

        axis = p[HEAD] - p[ABD]
        fwd = axis / (np.linalg.norm(axis) or 1)
        side = body_axes(fly.yaw)[2]
        side = side - fwd * (side @ fwd)
        side = side / (np.linalg.norm(side) or 1)
        up = np.cross(side, fwd)
        Rb = np.stack([fwd, up, side], 1)
        fa = p[THX] - p[ABD]
        fa = fa / (np.linalg.norm(fa) or 1)
        Ra = np.stack([fa, np.cross(side, fa), side], 1)
        leg_col = col((86, 58, 26))
        for k in range(6):
            sgn = 1 if k < 3 else -1
            hip = p[THX] + fwd * (14 - 12 * (k % 3)) * S - up * 10 * S + side * sgn * 8 * S
            rd.add("cylinder", segment(hip, p[KNEE[k]], 2.6 * S), leg_col)
            rd.add("sphere", trs(p[KNEE[k]], None, (2.7 * S,) * 3), leg_col)
            rd.add("cylinder", segment(p[KNEE[k]], p[FOOT[k]], 1.8 * S), leg_col)
            rd.add("sphere", trs(p[FOOT[k]], None, (2.4 * S,) * 3), col((60, 40, 18)))
        rd.add("sphere", trs(p[ABD], Ra, (30 * S * k_, 21 * S * k_, 21 * S * k_)), col((176, 116, 44)), P_STRIPES)
        rd.add("sphere", trs(p[THX], Rb, (23 * S * k_, 19 * S * k_, 19 * S * k_)), col((156, 102, 40)))
        rd.add("sphere", trs(p[HEAD], Rb, (15 * S * k_,) * 3), col((146, 96, 38)))
        for sgn in (1, -1):
            eye = p[HEAD] + fwd * 3 * S + up * 2 * S + side * sgn * 9 * S * k_
            rd.add("sphere", trs(eye, Rb, (8 * S * k_, 11 * S * k_, 9 * S * k_)), col((196, 30, 26)), P_EYE)
            if dead:
                for a in (0.8, -0.8):
                    rd.add("cube", trs(eye + side * sgn * 8 * S, Rb @ rot_z(a), (18 * S, 3 * S, 3 * S)), (0.12, 0.08, 0.08))
            else:
                rd.add("sphere", trs(eye + up * 5 * S + fwd * 4 * S + side * sgn * 6 * S, None, (2.4 * S,) * 3), (1, 0.85, 0.8), P_NONE, 0.6)
            a0 = p[HEAD] + fwd * 11 * S + up * 8 * S + side * sgn * 4 * S
            rd.add("cylinder", segment(a0, a0 + fwd * 10 * S + up * 10 * S + side * sgn * 5 * S, 1.2 * S), col((70, 45, 20)))
        rd.add("cylinder", segment(p[HEAD] + fwd * 8 * S - up * 10 * S, p[HEAD] + fwd * 14 * S - up * 20 * S, 1.8 * S), col((70, 45, 20)))
        flap = not dead and (now < fly.escape_until or (fly.grabbed is not None and random.random() < 0.3))
        for i, sgn in ((0, 1), (1, -1)):
            base = p[THX] + up * 12 * S + side * sgn * 5 * S
            tip = p[WING[i]]
            if flap:
                tip = base + (tip - base) * 0.8 + up * 38 * S * math.sin(now * 90 + i) + side * sgn * 10 * S
            span = float(np.linalg.norm(tip - base))
            rd.add("sphere", trs((base + tip) / 2, frame_from_x(tip - base), (span / 2 + 6 * S, 1.2 * S, 13 * S)),
                   (0.82, 0.87, 0.95, 0.38 if not fly.melt else 0.2), P_NONE, 0.15)
        # overlays on the body
        if fly.soak > 0.05:
            rng = random.Random(int(now * 8))
            for _ in range(int(4 + 10 * fly.soak)):
                q = p[rng.choice((HEAD, THX, ABD, ABD))] + np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.12), rng.uniform(-0.12, 0.12)])
                rd.particle(q, rng.uniform(0.012, 0.03), (0.95, 0.97, 0.92, 0.7))
        if fly.frozen_at is not None:
            lo, hi = p.min(0) - 0.08, p.max(0) + 0.08
            rd.add("cube", trs((lo + hi) / 2, None, hi - lo), (0.72, 0.87, 0.97, 0.35), P_ICE, 0.1)
        elif fly.frost > 0.1:
            rng = random.Random(7)
            for _ in range(int(14 * fly.frost)):
                q = p[rng.choice((HEAD, THX, ABD, ABD))] + np.array([rng.uniform(-0.13, 0.13), rng.uniform(-0.1, 0.12), rng.uniform(-0.13, 0.13)])
                rd.add("cube", trs(q, rot_y(rng.uniform(0, 3)) @ rot_x(0.6), (0.02, 0.02, 0.02)), (0.9, 0.97, 1.0), P_ICE, 0.4)
        if fly.wrapped:
            mid = (p[HEAD] + p[ABD]) / 2
            L = float(np.linalg.norm(axis))
            rd.add("sphere", trs(mid, Rb, (L / 2 + 0.14, 0.15, 0.15)), (0.94, 0.94, 0.9, 0.78))
            for kk in range(-3, 4):
                rd.add("torus", trs(mid + fwd * kk * 0.06, Rb @ rot_z(math.pi / 2), (0.155, 0.155, 0.155)), (0.98, 0.98, 0.95), P_NONE, 0.2)
        if now < fly.burn_until:
            for i in (HEAD, THX, ABD):
                rd.particle(p[i] + np.random.uniform(-0.08, 0.08, 3) + (0, 0.06, 0), random.uniform(0.04, 0.09),
                            (1.0, random.uniform(0.45, 0.8), 0.15, 0.8), additive=True)
        if dead:
            e = min(1.0, (now - fly.dead_at) / 2.0)
            rd.add("torus", trs(p[HEAD] + (0, 0.2 + 0.06 * e, 0), None, (0.09, 0.09, 0.09)), (1.0, 0.85, 0.35), P_NONE, 1.5)
        elif now < fly.stun_until:
            for kk in range(4):
                a = now * 6 + kk * math.pi / 2
                rd.add("sphere", trs(p[HEAD] + (0.16 * math.cos(a), 0.18 + 0.03 * math.sin(a * 2), 0.16 * math.sin(a)), None, (0.022,) * 3),
                       (1.0, 0.84, 0.25), P_NONE, 1.2)
        if now < fly.eating_until:
            for kk in range(3):
                ph = (now * 0.9 + kk / 3) % 1.0
                rd.particle(p[HEAD] + ((kk - 1) * 0.1, 0.15 + 0.3 * ph, 0), 0.04, (1.0, 0.45, 0.6, 1 - ph), additive=True)
        self._shadow(rd, p[THX], 0.34)

    def _shadow(self, rd: Renderer, pos, radius: float) -> None:
        h = float(pos[1])
        k = float(np.clip(1 - h / 3.0, 0.15, 1))
        y = WATER3 + 0.004 if k2.ARENAS[self.arena_i] == "pool" and h > WATER3 else 0.008
        rd.add("cylinder", trs((pos[0], y, pos[2]), None, (radius * (1.2 - 0.5 * k), 0.002, radius * (1.2 - 0.5 * k) * 0.8)),
               (0.0, 0.0, 0.0, 0.35 * k), layer="blend")

    def _draw_extras(self, rd: Renderer, now: float) -> None:
        for b in self.bombs3:
            rd.add("sphere", trs(b["p"], None, (0.08, 0.08, 0.08)), (0.12, 0.12, 0.14))
            rd.add("cylinder", segment(b["p"] + (0, 0.06, 0), b["p"] + (0.03, 0.13, 0), 0.01), (0.7, 0.6, 0.4))
            if int(now * 12) % 2:
                rd.particle(b["p"] + (0.03, 0.14, 0), 0.04, (1.0, 0.85, 0.3, 1.0), additive=True)
            self._shadow(rd, b["p"], 0.09)
        for s in self.sugars3:
            e = max(0.35, s["left"]) * 0.07
            rd.add("cube", trs(s["p"], rot_y(0.5), (e, e, e)), (0.98, 0.98, 1.0), P_NONE, 0.1)
        if self.spider3 is not None:
            sp = self.spider3
            x, y, z = sp["p"]
            rd.add("cylinder", segment((x, y + 0.05, z), (x, RY, z), 0.002), (0.9, 0.9, 0.92, 0.7))
            body = np.array([x, y, z])
            black = (0.12, 0.11, 0.13)
            rd.add("sphere", trs(body, None, (0.075, 0.06, 0.09)), black)
            rd.add("sphere", trs(body + (0, 0.01, 0.09), None, (0.045, 0.04, 0.045)), black)
            for sgn in (-1, 1):
                for kk in range(4):
                    a = (kk - 1.5) * 0.45
                    wig = 0.015 * math.sin(now * 14 + kk * 1.7 + sgn)
                    knee = body + (sgn * 0.12 * math.cos(a), 0.06 + wig, 0.12 * math.sin(a))
                    foot = body + (sgn * 0.2 * math.cos(a), -0.1 - wig, 0.2 * math.sin(a))
                    rd.add("cylinder", segment(body, knee, 0.008), black)
                    rd.add("cylinder", segment(knee, foot, 0.006), black)
            for dx in (-0.015, 0.015):
                rd.add("sphere", trs(body + (dx, 0.03, 0.13), None, (0.008,) * 3), (1.0, 0.25, 0.25), P_NONE, 1.0)
            self._shadow(rd, body, 0.15)
        for sh in self.shards3:
            R = rot_x(sh["rot"][0]) @ rot_y(sh["rot"][1]) @ rot_z(sh["rot"][2])
            rd.add("cube", trs(sh["p"], R, (sh["size"], sh["size"] * 0.6, sh["size"] * 0.3)),
                   (0.55, 0.36, 0.18) if sh["fly"] else (0.78, 0.9, 1.0, 0.6), P_ICE)
        for a, b, t0 in self.bolts3:
            pts = [a + (b - a) * j / 7 + (np.random.normal(0, 0.05, 3) if 0 < j < 7 else 0) for j in range(8)]
            for j in range(7):
                rd.add("cylinder", segment(pts[j], pts[j + 1], 0.012), (0.85, 0.93, 1.0), P_NONE, 3.0)
                rd.particle(pts[j], 0.1, (0.5, 0.7, 1.0, 0.25), additive=True)
        for pt in self.parts:
            e = (now - pt["t"]) / pt["life"]
            kind = pt["kind"]
            if kind == "flame":
                c = (1.0, 0.95, 0.65) if e < 0.25 else (1.0, 0.6, 0.15) if e < 0.55 else (0.85, 0.25, 0.1)
                rd.particle(pt["p"], 0.03 + 0.12 * e, c + (0.9 * (1 - e ** 2),), additive=True)
            elif kind == "fire":
                rd.particle(pt["p"], pt["size"] * (0.5 + e), (1.0, 0.55 - 0.3 * e, 0.12, 0.8 * (1 - e)), additive=True)
            elif kind == "cleaner":
                rd.particle(pt["p"], 0.04 + 0.16 * e, (0.8, 0.9, 0.98, 0.35 * (1 - e)))
            elif kind == "freeze":
                rd.particle(pt["p"], 0.04 + 0.16 * e, (0.9, 0.97, 1.0, 0.5 * (1 - e)))
            elif kind == "dust":
                rd.particle(pt["p"], pt["size"] * (0.6 + e), (0.75, 0.7, 0.65, 0.45 * (1 - e)))
            elif kind == "streak":
                rd.particle(pt["p"], 0.025, (0.9, 0.95, 1.0, 0.35 * (1 - e)))

    def draw_viewmodel(self, rd: Renderer, now: float) -> None:
        """The tool in your hand, in camera space (x right, y up, -z forward)."""
        name = TOOLS[self.tool][0]
        bob = 0.012 * math.sin(self.player.walk_phase * 2)
        base = np.array([0.27, -0.26 + bob, -0.55])
        skin = (0.93, 0.74, 0.6)

        def hand(pos, curl: float, point: bool = False, flick: float = 0.0):
            rd.add("sphere", trs(pos, None, (0.045, 0.028, 0.06)), skin, layer="view")
            for j, dx in enumerate((-0.03, -0.01, 0.01, 0.03)):
                if point and j == 1:
                    ang = -0.2 - 1.2 * flick
                    tip = pos + (dx, 0.01 + 0.08 * math.sin(-ang) * 0.4, -0.05 - 0.07 * math.cos(ang))
                else:
                    tip = pos + (dx, -0.02 * curl, -0.05 - 0.05 * (1 - curl))
                rd.add("cylinder", segment(pos + (dx, 0, -0.04), tip, 0.009), skin, layer="view")
            rd.add("cylinder", segment(pos + (-0.04, 0, -0.01), pos + (-0.07, 0.01, -0.05), 0.01), skin, layer="view")

        if name == "hand":
            hand(base + (0, 0.02, 0), 0.9 if self.fly.grabbed is not None else 0.1)
        elif name == "flick":
            hand(base + (0, 0.02, 0), 0.9, point=True, flick=max(0.0, 1 - (now - self.flick_t) / 0.15))
        elif name == "swatter":
            ph = now - self.swing_t
            ang = -0.9 * (1 - (ph / 0.12)) + 0.2 if ph < 0.12 else 0.2 - 0.2 * min(1.0, (ph - 0.12) / 0.3) if ph < 0.42 else 0.0
            R = rot_x(-ang - 0.3)
            pivot = base + (0, -0.05, 0.05)
            handle_end = pivot + R @ np.array([0.0, 0.32, -0.12])
            rd.add("cylinder", segment(pivot, handle_end, 0.012), (0.5, 0.33, 0.18), layer="view")
            head = handle_end + R @ np.array([0.0, 0.1, -0.03])
            rd.add("cube", trs(head, R, (0.2, 0.22, 0.012)), (0.82, 0.16, 0.18, 0.9), layer="view_blend")
            hand(pivot + (0, 0.02, 0.02), 1.0)
        elif name == "bomb":
            if now - self.throw_t > 0.4:
                rd.add("sphere", trs(base + (0, 0.05, -0.05), None, (0.07, 0.07, 0.07)), (0.12, 0.12, 0.14), layer="view")
            hand(base, 0.7)
        elif name == "torch":
            rd.add("cylinder", segment(base + (0, 0, 0.05), base + (0, 0.06, -0.2), 0.035), (0.75, 0.2, 0.18), layer="view")
            rd.add("cylinder", segment(base + (0, 0.06, -0.2), base + (0, 0.07, -0.26), 0.012), (0.6, 0.6, 0.65), layer="view")
            hand(base + (0, -0.02, 0.04), 1.0)
        elif name in ("cleaner", "freeze"):
            body = (0.82, 0.14, 0.14) if name == "cleaner" else (0.25, 0.5, 0.85)
            rd.add("cylinder", segment(base + (0, -0.05, 0), base + (0, 0.12, -0.02), 0.04), body, layer="view")
            rd.add("cylinder", segment(base + (0, 0.02, -0.01), base + (0, 0.06, -0.012), 0.041), (0.95, 0.95, 0.96), layer="view")
            rd.add("cylinder", segment(base + (0, 0.12, -0.02), base + (0, 0.15, -0.06), 0.012), (0.2, 0.2, 0.22), layer="view")
            hand(base + (0, -0.03, 0.03), 1.0)
        elif name == "zapper":
            rd.add("cylinder", segment(base, base + (0, 0.12, -0.12), 0.014), (0.2, 0.2, 0.25), layer="view")
            ring = base + (0, 0.2, -0.2)
            rd.add("torus", trs(ring, rot_x(-0.7), (0.09, 0.09, 0.09)), (0.95, 0.85, 0.2), P_NONE,
                   2.0 if now - self.zap_ready > -0.3 and now < self.zap_ready else 0.3, layer="view")
            hand(base + (0, 0.0, 0.02), 1.0)
        elif name == "spider":
            rd.add("sphere", trs(base + (0, 0.06, -0.05), None, (0.04, 0.035, 0.05)), (0.12, 0.11, 0.13), layer="view")
            hand(base, 0.6)
        elif name == "sugar":
            if now - self.throw_t > 0.3:
                rd.add("cube", trs(base + (0, 0.05, -0.05), rot_y(0.5), (0.055, 0.055, 0.055)), (0.98, 0.98, 1.0), layer="view")
            hand(base, 0.6)

    # --- HUD --------------------------------------------------------------------------------------------------------------------------
    def draw_hud3d(self, now: float, project) -> None:
        hud = self.screen
        hud.fill((0, 0, 0, 0))
        for pos, text, t0, color in self.popups3:
            sp = project(pos)
            if sp is None:
                continue
            e = (now - t0) / 0.9
            txt = self.f_big.render(text, True, color)
            shd = self.f_big.render(text, True, (120, 16, 22))
            a = int(255 * (1 - e ** 3))
            txt.set_alpha(a)
            shd.set_alpha(a)
            x, y = sp[0] - txt.get_width() / 2, sp[1] - 50 * e - 20
            hud.blit(shd, (x + 3, y + 3))
            hud.blit(txt, (x, y))
        if not self._overlay_open():
            cx, cy = k2.PLAY_W // 2, self.hud_h // 2
            eye, d = self.aim()
            i, t, _ = self.fly.nearest_to_ray(eye, d, REACH, 0.25)
            col = (255, 190, 90) if i is not None else (240, 240, 240)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pygame.draw.line(hud, (0, 0, 0, 120), (cx + dx * 5 + 1, cy + dy * 5 + 1), (cx + dx * 13 + 1, cy + dy * 13 + 1), 3)
                pygame.draw.line(hud, col, (cx + dx * 5, cy + dy * 5), (cx + dx * 13, cy + dy * 13), 2)
        self._draw_toolbar(hud)
        self._draw_hud(hud, now)
        if not self.look and not self._overlay_open():
            msg = self.f_bold.render("click the room (or press Tab) to look around" if not self.quit_armed else
                                     "press Esc again to quit, or click the room to keep playing", True, INK_ON)
            box = msg.get_rect(center=(k2.PLAY_W // 2, self.hud_h // 2 + 60)).inflate(24, 12)
            pygame.draw.rect(hud, (8, 10, 16, 200), box, border_radius=8)
            hud.blit(msg, msg.get_rect(center=box.center))
        if self.report is not None:
            self._draw_autopsy(hud, now)
        elif self.big_view:
            self._draw_big_view(hud)
        if self.surgery_open:
            self._draw_surgery(hud)
        if self.help_open:
            self._draw_help(hud)
        if PANEL_MODES[self.panel_mode][0] != "hidden":
            self._draw_brain(now)
        else:
            self.view_rect = pygame.Rect(0, 0, 0, 0)

    def _draw_help(self, surf) -> None:
        panel = pygame.Rect(135, 80, 620, 64 + 30 * len(HELP3D))
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, k2.BORDER, panel, 1, border_radius=16)
        self._text(surf, "CONTROLS", (panel.x + 24, panel.y + 16), k2.INK, self.f_head)
        for kk, (key, what) in enumerate(HELP3D):
            y = panel.y + 54 + kk * 30
            self._text(surf, key, (panel.x + 30, y), k2.AMBER, self.f_bold)
            self._text(surf, what, (panel.x + 190, y), k2.TEXT, self.f_text)

    # --- input ------------------------------------------------------------------------------------------------------------------------
    def set_look(self, on: bool) -> None:
        self.look = on
        self.quit_armed = False
        pygame.event.set_grab(on)
        pygame.mouse.set_visible(not on)
        pygame.mouse.get_rel()

    def handle3d(self, ev, now: float, to_logical) -> bool:
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                if self.look:
                    self.set_look(False)
                    return True
                if self._overlay_open():
                    self.help_open = self.surgery_open = self.big_view = False
                    return True
                if self.quit_armed:
                    return False
                self.quit_armed = True
                return True
            if ev.key == pygame.K_TAB:
                self.set_look(not self.look)
                return True
            if ev.key == pygame.K_F12:
                self.save_png()
                return True
            if ev.key == pygame.K_v:
                self.panel_mode = (self.panel_mode + 1) % len(PANEL_MODES)
                self.panel_alpha = PANEL_MODES[self.panel_mode][1]
                self.saved_msg = (f"brain panel: {PANEL_MODES[self.panel_mode][0]}", time.perf_counter())
                return True
            if ev.key == pygame.K_u:
                self.ui_mode = UI_MODES[(UI_MODES.index(self.ui_mode) + 1) % len(UI_MODES)]
                self.saved_msg = (f"menu size: {self.ui_mode}", time.perf_counter())
                return True
            if ev.key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_LSHIFT, pygame.K_LCTRL, pygame.K_c, pygame.K_SPACE):
                return True
            if ev.key == pygame.K_r and self.report is not None:
                self.new_fly()
                return True
            return k2.Game.handle(self, ev, now)
        if ev.type == pygame.MOUSEWHEEL and self.look:
            self.tool = (self.tool - ev.y) % len(TOOLS)
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.look:
                self.use_tool3d(now)
                return True
            pos = to_logical(ev.pos)
            if not self._overlay_open() and pos[0] < k2.PLAY_W and not any(r.collidepoint(pos) for r in getattr(self, "tool_rects", [])):
                self.set_look(True)
                return True
            if self.surgery_open or self.help_open or self.report is not None or self.big_view:
                return k2.Game.handle(self, pygame.event.Event(ev.type, button=1, pos=pos), now)
            for kk, r in enumerate(getattr(self, "tool_rects", [])):
                if r.collidepoint(pos):
                    self.tool = kk
                    return True
            if self.view_rect.collidepoint(pos):
                self.big_view = True
            return True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if not self.fly.wrapped:
                self.fly.grabbed = None
            self.torching = False
            return True
        return True

    def capture(self) -> None:                     # frames are captured by the 3D app (see App.capture)
        pass

    def save_png(self) -> None:
        self.want_png = True


INK_ON = (240, 243, 248)


class App:
    """Window, GL context, frame composition, fullscreen and scaling."""

    def __init__(self, fullscreen: bool):
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((k2.W, k2.H), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption("Kick the Fly")
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.rd = Renderer(self.ctx)
        desk = pygame.display.get_desktop_sizes()[0] if pygame.display.get_desktop_sizes() else (k2.W, k2.H)
        if fullscreen or desk[0] < k2.W or desk[1] < k2.H + 60:
            pygame.display.toggle_fullscreen()
        self.hud_tex = None
        self.hud_size = None
        self.scene_size = None
        self.small = self.ctx.simple_framebuffer(k2.GIF_SIZE)

    def layout(self, game=None, size=None):
        Wn, Hn = size or pygame.display.get_window_size()
        ui, panel = (game.ui_mode, game.panel_mode) if game is not None else ("crisp", 0)
        s, hud_w, hud_h, play_w, view_w = compute_layout(Wn, Hn, ui, panel)
        return Wn, Hn, s, hud_w, hud_h, play_w, view_w

    def hud_texture(self, hud_w: int, hud_h: int, s: float) -> moderngl.Texture:
        if self.hud_size != (hud_w, hud_h):
            self.hud_size = (hud_w, hud_h)
            self.hud_tex = self.ctx.texture((hud_w, hud_h), 4)
        crisp = abs(s - round(s)) < 1e-6
        self.hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST) if crisp else (moderngl.LINEAR, moderngl.LINEAR)
        return self.hud_tex

    def _ensure_scene(self, w: int, h: int):
        if self.scene_size == (w, h):
            return
        self.scene_size = (w, h)
        samples = min(4, self.ctx.max_samples)
        try:
            self.ms = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((w, h), 4, samples=samples)],
                                           depth_attachment=self.ctx.depth_renderbuffer((w, h), samples=samples))
        except Exception:
            self.ms = None
        self.scene_tex = self.ctx.texture((w, h), 4)
        self.scene_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.scene = self.ctx.framebuffer(color_attachments=[self.scene_tex], depth_attachment=self.ctx.depth_texture((w, h)))

    def render(self, game: Game3D, now: float, target=None, size=None):
        ctx, rd = self.ctx, self.rd
        Wn, Hn, s, hud_w, hud_h, play_w, view_w = self.layout(game, size)
        k2.W, k2.H, k2.PLAY_W, k2.FLOOR = hud_w, hud_h, play_w, hud_h - 120   # the shared HUD code reads these
        game.view_w, game.hud_h = view_w, hud_h
        if game.screen.get_size() != (hud_w, hud_h):
            game.screen = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        vw, vh = max(1, int(round(view_w * s))), Hn
        self._ensure_scene(vw, vh)
        fb = self.ms or self.scene
        fb.use()
        ctx.viewport = (0, 0, vw, vh)
        ctx.clear(*SKY_CLEAR, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        pl = game.player
        eye = pl.eye
        shake = np.random.uniform(-0.01, 0.01, 3) if now < game.shake_until else 0
        f, r, u = pl.basis()
        view = look_at(eye + shake, eye + shake + f)
        proj = perspective(math.radians(70), vw / vh, 0.03, 40.0)
        # when the 3D view runs under a see-through panel, shift the lens so the crosshair and your hand stay centered
        # on the open part of the screen
        lens = np.eye(4)
        lens[0, 3] = play_w / view_w - 1                  # shift in clip x by w: moves the image center left
        proj = lens @ proj
        lamp_on = k2.ARENAS[game.arena_i] == "lamp"
        lights = dict(u_sun_dir=np.array([0.3, -0.55, 0.78]) / np.linalg.norm([0.3, -0.55, 0.78]), u_sun_col=(0.95, 0.88, 0.75),
                      u_sky=(0.42, 0.44, 0.5), u_ground=(0.24, 0.2, 0.17), u_lp0=(0.0, RY - 0.3, 0.0), u_lc0=(2.4, 2.2, 1.9),
                      u_lp1=tuple(LAMP3), u_lc1=(3.5, 2.8, 1.8) if lamp_on else (0, 0, 0))
        rd.clear()
        game.draw_world(rd, now)
        rd.set_scene(view, proj, eye, lights, now)
        rd.draw_layer("opaque")
        rd.draw_layer("blend")
        rd.draw_particles()
        # the tool in your hand: squeezed into the front 10% of the depth range so it never sinks into walls,
        # lit by the same lights moved into camera space
        game.draw_viewmodel(rd, now)
        Rv = view[:3, :3]
        cam_lights = dict(lights)
        cam_lights["u_sun_dir"] = Rv @ lights["u_sun_dir"]
        cam_lights["u_lp0"] = (view @ np.append(lights["u_lp0"], 1))[:3]
        cam_lights["u_lp1"] = (view @ np.append(lights["u_lp1"], 1))[:3]
        squeeze = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.1, -0.9], [0, 0, 0, 1.0]])
        rd.set_scene(np.eye(4), squeeze @ lens @ perspective(math.radians(60), vw / vh, 0.01, 5.0), (0, 0, 0), cam_lights, now)
        rd.draw_layer("view")
        rd.draw_layer("view_blend")
        if self.ms is not None:
            ctx.copy_framebuffer(self.scene, self.ms)

        def project(p):
            c = proj @ view @ np.append(p, 1.0)
            if c[3] <= 0.05:
                return None
            ndc = c[:3] / c[3]
            if abs(ndc[0]) > 1.2 or abs(ndc[1]) > 1.2:
                return None
            return ((ndc[0] + 1) / 2 * view_w, (1 - ndc[1]) / 2 * hud_h)

        game.draw_hud3d(now, project)
        tex = self.hud_texture(hud_w, hud_h, s)
        tex.write(pygame.image.tobytes(game.screen, "RGBA", False))
        out = target or ctx.screen
        out.use()
        ctx.viewport = (0, 0, Wn, Hn)
        out.clear(0, 0, 0, 1)
        rd.blit_texture(self.scene_tex, (0, 0, vw, vh), (Wn, Hn), flip=False, blend=False)
        rd.blit_texture(tex, (0, 0, hud_w * s, hud_h * s), (Wn, Hn), flip=True, blend=True)
        self.view_frac = view_w / hud_w
        return (Wn, Hn, s, hud_w, hud_h, play_w, view_w)

    def capture(self, game: Game3D) -> None:
        """Downscale the finished frame into the GIF buffer (15 fps)."""
        w, h = k2.GIF_SIZE
        self.small.use()
        self.ctx.viewport = (0, 0, w, h)
        self.small.clear(0, 0, 0, 1)
        self.scene_tex.build_mipmaps()
        self.scene_tex.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
        self.hud_tex.build_mipmaps()
        self.hud_tex.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
        self.rd.blit_texture(self.scene_tex, (0, 0, w * getattr(self, "view_frac", 0.7), h), (w, h), flip=False, blend=False)
        self.rd.blit_texture(self.hud_tex, (0, 0, w, h), (w, h), flip=True, blend=True)
        self.scene_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.hud_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.hud_size = None                            # restore the right filter next frame
        data = self.small.read(components=3)
        rows = np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1]
        game.frames.append(rows.tobytes())

    def screenshot(self, path: Path, lay) -> None:
        Wn, Hn = lay[0], lay[1]
        data = self.ctx.screen.read(viewport=(0, 0, Wn, Hn), components=3)   # the screen's own size is stale after F11
        img = pygame.image.frombytes(data, (Wn, Hn), "RGB", True)
        pygame.image.save(img, str(path))


def run(smoke: float = 0.0, shot: str | None = None, fullscreen: bool = False) -> int:
    app = App(fullscreen)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui,consolas", 22)
    state: dict = {"stage": "starting"}
    threading.Thread(target=k2.load_brain, args=(state,), daemon=True).start()
    t0 = time.perf_counter()
    splash = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    while "brain" not in state:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return 0
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()
        splash.fill((9, 11, 15, 255))
        msg = state.get("error") or f"{state['stage']}{'.' * (int((time.perf_counter() - t0) * 3) % 4)}"
        img = font.render(msg, True, k2.RED if "error" in state else k2.TEXT)
        splash.blit(img, img.get_rect(center=(k2.W // 2, k2.H // 2)))
        Wn, Hn = pygame.display.get_window_size()
        s = min(Wn / k2.W, Hn / k2.H)
        tex = app.hud_texture(k2.W, k2.H, 1.5)
        tex.write(pygame.image.tobytes(splash, "RGBA", False))
        app.ctx.screen.use()
        app.ctx.viewport = (0, 0, Wn, Hn)
        app.ctx.screen.clear(0, 0, 0, 1)
        app.rd.blit_texture(tex, ((Wn - k2.W * s) / 2, (Hn - k2.H * s) / 2, k2.W * s, k2.H * s), (Wn, Hn), flip=True)
        pygame.display.flip()
        clock.tick(30)

    brain = state["brain"]
    brain.start()
    hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    game = Game3D(hud, brain, state["view"])
    game.want_png = False
    running = True
    t_game = last = time.perf_counter()
    lay = app.layout(game)

    def to_logical(pos):
        return (int(pos[0] / lay[2]), int(pos[1] / lay[2]))

    while running:
        now = time.perf_counter()
        dt = min(0.05, now - last)
        last = now
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and (ev.key == pygame.K_F11 or (ev.key == pygame.K_RETURN and ev.mod & pygame.KMOD_ALT)):
                pygame.display.toggle_fullscreen()
                continue
            running = game.handle3d(ev, now, to_logical) and running
        if game.look and game._overlay_open():
            game.set_look(False)
        rel = pygame.mouse.get_rel() if game.look else (0, 0)
        kp = pygame.key.get_pressed()
        keys = dict(w=kp[pygame.K_w], s=kp[pygame.K_s], a=kp[pygame.K_a], d=kp[pygame.K_d],
                    sprint=kp[pygame.K_LSHIFT] or kp[pygame.K_RSHIFT], crouch=kp[pygame.K_LCTRL] or kp[pygame.K_c])
        game.update3d(now, dt, keys, rel)
        lay = app.render(game, now)
        if game.frame % 4 == 0:
            app.capture(game)
        if game.want_png:
            game.want_png = False
            path = game._save_dir() / f"kick-the-fly-{time.strftime('%Y%m%d-%H%M%S')}.png"
            app.ctx.screen.use()
            app.screenshot(path, lay)
            game.saved_msg = (f"saved {path}", time.perf_counter())
        pygame.display.flip()
        if smoke and now - t_game > smoke:
            try:
                from PIL import Image  # noqa: F401
                gif = "gif ok"
            except ImportError:
                gif = "no gif"
            status = f"smoke ok 3d: {brain.n:,} neurons, {brain.steps_per_s:.0f} steps/s, {clock.get_fps():.0f} fps, sound {game.sound.ok}, {gif}"
            print(status)
            if shot:
                app.ctx.screen.use()
                app.screenshot(Path(shot), lay)
                with open(shot + ".txt", "w") as fh:
                    fh.write(status)
            break
        clock.tick(60)
    brain.stop()
    pygame.quit()
    return 0
