"""Kick the Fly: a kick-the-buddy toy whose buddy is the MaleCNS v1.0 connectome.

Every hit lands on real touch-sensing neurons, and the live LIF sim from
connectome/sim.py decides how the fly reacts:

  where you hit          sensory neurons driven
  head                   head bristles (BM_*) + Johnston's organ (JO-*)
  thorax / abdomen       tactile neurons SNta*
  legs                   proprioceptive neurons SNpp*
  wings                  wing sensory neurons WG*

  descending neurons                   what the fly does
  DNg85 DNg48 DNg37 DNge067 DNg29 ...  jump away
  DNge122 DNge104 DNge102 DNxl114 ...  run away
  DNge074 DNge075 DNge096 DNg30 ...    kick its legs
  DNp09 (P9, forward walking)          walk
  MDN (moonwalker)                     back up
  DNa01 + DNa02, right minus left      turn around

The three touch groups are the DN types that responded most to each touch site
when the sim was probed (40 hits per site, 400 ms before vs after). Which move
each group triggers is a game choice. The giant fiber DNp01 barely reacts to
touch in this model, so it isn't used. Wing touches mostly raise DNp09.

Brain view: a front view built from the neurons' real cell-body positions. Each
neuron is drawn as a fiber from its cell body toward the center of its synaptic
partners, colored by direction, and glows when it fires above its calm rate.
Real neuron shapes aren't bundled, so the fibers are estimates. B toggles a big view.

Tracking: every neuron's spikes are counted into its group each 5 ms step. Each
group's rate is compared with its own calm baseline, a 10 s average taken only
while nothing has touched the fly for 2 s, so hits don't inflate "normal".

Pain: the adult connectome has no neurons annotated as nociceptors, so the PAIN
index is a game estimate built from real signals, not a measurement of what the
fly feels:
  55%  touch overload  all touch neurons above calm, saturating at +35 spikes/s
  25%  heat sensors    thermosensory TRN_VP1m/VP2/VP3a above calm, saturating at 55
  20%  DN alarm        all descending neurons above calm, saturating at +30%,
                       held at its peak for ~1 s
The blowtorch drives every touch and heat neuron at full strength, which pins it.

Brake cleaner: the spray drives the fly's smell neurons (ORN_*, 2,635) and taste
neurons (leg, labellar and pharyngeal GRNs) at full strength. Solvents like this
depress nervous systems, so as the fly dissolves an inhibitory current grows on
every neuron (game rule; its strength is not measured) and activity fades before
it melts into a puddle.

Death: health runs out from hits. A sim can't die on its own, so on death the
tonic drive is cancelled with an inhibitory ramp over 1.5 s and the global gain
controller is frozen. The autopsy compares each population's last 2 s alive
with its calm baseline.

Also game rules, not the connectome: the ragdoll physics, standing back up,
the direction of jumps and runs (away from the last hit), being stunned, damage.

    .venv\\Scripts\\python.exe kick_the_fly.py
"""
from __future__ import annotations

import math
import random
import sys
import threading
import time

import numpy as np
import pygame
import scipy.sparse as sp
from pygame import gfxdraw

W, H = 1280, 760
PLAY_W = 890                 # left part is the arena, right part the brain panel
FLOOR = 640
CEIL = 40
BG = (9, 11, 15)
PANEL_BG = (13, 16, 22)
CARD = (20, 24, 32)
BORDER = (40, 47, 60)
LABEL = (120, 132, 150)
TEXT = (205, 212, 224)
INK = (240, 243, 248)
DIM = (70, 78, 92)
ACCENT = (86, 214, 255)
AMBER = (255, 176, 64)
RED = (235, 70, 60)
UP = (230, 103, 103)         # diverging pair (red <-> blue) with a gray midpoint
DOWN = (57, 135, 229)
MID = (56, 56, 53)
S_GOOD, S_WARN, S_CRIT = (12, 163, 12), (250, 178, 25), (208, 59, 59)
SERIES = ((57, 135, 229), (217, 89, 38), (25, 158, 112))   # blue, orange, aqua

# --- brain -------------------------------------------------------------------
SENSE = {"head": ("BM_", "JO-"), "body": ("SNta",), "legs": ("SNpp",), "wing": ("WG",), "heat": ("TRN_",),
         "smell": ("ORN_",), "taste": ("LgLG", "LgAG", "LB", "PhG", "claw_")}
TOUCH = ("head", "body", "legs", "wing")
MOTOR = (  # name, types, side, label
    ("jump", ("DNg85", "DNg48", "DNg37", "DNge067", "DNg29", "DNge132"), None, "head-touch DNs  > JUMP"),
    ("run", ("DNge122", "DNge104", "DNge102", "DNxl114", "DNge182", "DNge048"), None, "body-touch DNs  > RUN"),
    ("kick", ("DNge074", "DNge075", "DNge096", "DNg30", "DNg34", "DNp38"), None, "leg-touch DNs   > KICK"),
    ("walk", ("DNp09",), None, "DNp09 (P9)      > WALK"),
    ("back", ("MDN",), None, "MDN moonwalker  > BACK UP"),
    ("turn_l", ("DNa01", "DNa02"), "L", "DNa01/02 left   > TURN"),
    ("turn_r", ("DNa01", "DNa02"), "R", "DNa01/02 right  > TURN"),
)
POPS = (  # population name, superclasses
    ("photoreceptors", ("ol_sensory",)),
    ("optic lobe", ("ol_intrinsic", "visual_projection", "visual_centrifugal", "visual_projection_tbc")),
    ("central brain", ("cb_intrinsic",)),
    ("head sensory", ("cb_sensory", "cb_sensory_tbc", "sensory_descending")),
    ("body sensory", ("vnc_sensory", "vnc_sensory_tbc", "sensory_ascending", "sensory_ascending_tbc")),
    ("ascending", ("ascending_neuron",)),
    ("descending", ("descending_neuron", "descending_neuron_tbc")),
    ("nerve cord", ("vnc_intrinsic", "vnc_tbc")),
    ("motor neurons", ("vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "efferent_ascending", "efferent_descending")),
)
# x calm baseline. Probed over 60 s with no touch the maxima were jump 2.6, run 1.8, kick 1.7; typical hits reach
# jump 4.6 (head), run 2.9-3.8 (body), kick 2.0-2.8 (legs). walk/back/turn sit between their spontaneous p99 and
# p99.9 so the fly wanders on its own every so often.
THRESH = {"jump": 3.0, "run": 2.4, "kick": 2.0, "walk": 3.0, "back": 3.8, "turn": 2.1}
STIM_AMP = 0.5               # x ext_gain 4 = 2.0 per step: a driven neuron fires every refractory cycle
HIST = 1500                  # history samples, one per 20 ms = 30 s
CALM_STEPS = 400             # 2 s without a touch before the baseline learns again


class Brain:
    """Runs the LIF sim in real time on its own thread; hits queue sensory current, the game reads group rates."""

    def __init__(self, g, sim, seed: int = 0):
        self.sim, self.n = sim, g.n
        self.dt = sim.p.dt_ms / 1000.0
        types = g.type.astype(str)
        sc = g.superclass.astype(str)
        sides = np.array([(i or "")[-2:] for i in g.instance])
        is_dn = sc == "descending_neuron"
        self.rng = np.random.default_rng(seed)

        # detail groups (touch sites + reaction DNs) are disjoint; population groups cover the whole connectome
        self.det_id = np.full(g.n, -1, np.int32)
        self.pop_id = np.full(g.n, -1, np.int32)
        self.names: list[str] = []
        sizes: list[int] = []
        self.sense: dict[tuple[str, str | None], np.ndarray] = {}

        def add_detail(name, m):
            m = m & (self.det_id < 0)
            self.det_id[m] = len(self.names)
            self.names.append(name)
            sizes.append(int(m.sum()))
            return m

        for region, prefixes in SENSE.items():
            m = np.zeros(g.n, bool)
            for p in prefixes:
                m |= np.char.startswith(types, p)
            m = add_detail(region, m & (np.char.find(sc, "sensory") >= 0))
            self.sense[(region, None)] = np.flatnonzero(m)
            for s in "LR":
                rows = np.flatnonzero(m & (sides == f"_{s}"))
                self.sense[(region, s)] = rows if len(rows) > 0.2 * m.sum() else np.flatnonzero(m)
        for name, tys, side, _ in MOTOR:
            m = is_dn & np.isin(types, tys)
            if side:
                m &= sides == f"_{side}"
            add_detail(name, m)
        self.n_det = len(self.names)
        for k, (name, classes) in enumerate(POPS):
            m = np.isin(sc, classes)
            self.pop_id[m] = k
            self.names.append(name)
            sizes.append(int(m.sum()))
        self.names.append("whole brain")
        sizes.append(g.n)

        G = len(self.names)
        self.col = {n: i for i, n in enumerate(self.names)}
        self.g_size = np.maximum(np.array(sizes, float), 1)
        self.fast = np.zeros(G)                      # Hz per neuron, EMA ~150 ms
        self.base = np.zeros(G)                      # calm baseline, Hz per neuron
        self.k_fast = 1 - math.exp(-1 / 30)
        self.k_base = 1 - math.exp(-1 / 2000)
        self.hist = np.zeros((HIST, G), np.float32)
        self.hist_n = 0                              # samples written so far
        self.last_poke = -10**9
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str | None], list] = {}
        self._cur = np.zeros(g.n, np.float32)
        self._inhib = np.full(g.n, -0.4, np.float32)  # x ext_gain 4 = -1.6 per step, far past bias + noise
        self._stop = False
        self.death_step: int | None = None
        self.sedation = 0.0                          # 0..1 share of the death inhibition, set by the game
        self.death_sample = 0
        self.death_base: np.ndarray | None = None
        self._revive = False
        self._gain_adapt = sim.p.gain_adapt
        self._gain = sim.gain
        self.steps_per_s = 0.0
        self.steps = 0

    @property
    def dead(self) -> bool:
        return self.death_step is not None

    def poke(self, region: str, side: str | None, strength: float) -> None:
        """Drive a random share of a region's touch neurons; harder hits recruit more of them for longer."""
        if self.dead:
            return
        s = float(np.clip(strength, 0, 1))
        pop = self.sense[(region, side)]
        rows = self.rng.choice(pop, size=max(1, int(len(pop) * (0.3 + 0.7 * s))), replace=False)
        steps = 8 + int(40 * s)
        with self._lock:
            self.last_poke = self.steps
            old = self._pending.get((region, side))
            if old is None or steps > old[1]:
                self._pending[(region, side)] = [rows, steps]

    def kill(self) -> None:
        with self._lock:
            self._pending.clear()
            self._gain = self.sim.gain
            self.death_base = self.base.copy()
            self.death_sample = self.hist_n
            self.death_step = self.steps
        self.sim.p.gain_adapt = 0.0

    def revive(self) -> None:
        self._revive = True

    def _step(self) -> None:
        with self._lock:
            active = list(self._pending.items())
            for key, item in active:
                item[1] -= 1
                if item[1] <= 0:
                    del self._pending[key]
        for _, (rows, _) in active:
            self._cur[rows] = STIM_AMP
        if self.death_step is not None:
            ramp = min(1.0, (self.steps - self.death_step) / 300)
            spikes = self.sim.step(self._inhib * np.float32(ramp))
        elif self.sedation > 0:                      # solvent depression while dissolving
            spikes = self.sim.step(self._cur + self._inhib * np.float32(self.sedation))
        else:
            spikes = self.sim.step(self._cur if active else None)
        for _, (rows, _) in active:
            self._cur[rows] = 0

        on = np.flatnonzero(spikes)
        G = len(self.names)
        counts = np.zeros(G)
        d = self.det_id[on]
        counts[:self.n_det] = np.bincount(d[d >= 0], minlength=self.n_det)
        p = self.pop_id[on]
        counts[self.n_det:G - 1] = np.bincount(p[p >= 0], minlength=len(POPS))
        counts[-1] = len(on)
        inst = counts / self.g_size / self.dt
        self.fast += (inst - self.fast) * self.k_fast
        if self.death_step is None and self.sedation == 0 and self.steps - self.last_poke > CALM_STEPS:
            self.base += (self.fast - self.base) * self.k_base
        self.steps += 1
        if self.steps % 4 == 0:
            self.hist[self.hist_n % HIST] = self.fast
            self.hist_n += 1

    def warmup(self, steps: int = 600) -> None:
        k = self.k_base
        self.k_base = 1 - math.exp(-1 / 60)          # let the baseline settle quickly
        for _ in range(steps):
            self._step()
        self.k_base = k

    def start(self) -> None:
        threading.Thread(target=self._loop, name="brain", daemon=True).start()

    def stop(self) -> None:
        self._stop = True

    def _loop(self) -> None:
        t0 = time.perf_counter()
        done, last_t, last_steps = 0, t0, self.steps
        while not self._stop:
            if self._revive:
                self._revive = False
                self.death_step = None
                self.sim.p.gain_adapt = self._gain_adapt
                self.sim.gain = self._gain
                for _ in range(300):
                    self._step()
                t0, done = time.perf_counter(), 0
            now = time.perf_counter()
            due = int((now - t0) / self.dt)
            if due - done > 40:                      # fell behind: drop the backlog instead of racing
                done = due - 40
            if done >= due:
                time.sleep(0.001)
                continue
            self._step()
            done += 1
            if now - last_t > 1.0:
                self.steps_per_s = (self.steps - last_steps) / (now - last_t)
                last_t, last_steps = now, self.steps

    def level(self, name: str) -> float:
        i = self.col[name]
        return float(self.fast[i] / max(self.base[i], 2.0))

    def hz(self, name: str) -> float:
        return float(self.fast[self.col[name]])

    def pain_parts(self, rates: np.ndarray, base: np.ndarray) -> np.ndarray:
        """(touch overload, heat, DN alarm) in 0..1 for one or many rows of group rates. See the module docstring."""
        rates = np.atleast_2d(rates)
        idx = [self.col[n] for n in TOUCH]
        sz = self.g_size[idx]
        touch = ((rates[:, idx] - base[idx]) * sz).sum(1) / sz.sum() / 35.0
        h = self.col["heat"]                         # thermosensors fire ~24 spikes/s even at rest
        heat = (rates[:, h] - base[h]) / max(55.0 - float(base[h]), 10.0)
        d = self.col["descending"]
        alarm = (rates[:, d] / max(float(base[d]), 1.0) - 1.0) / 0.3
        return np.clip(np.stack([touch, heat, alarm], 1), 0, 1)

    @staticmethod
    def hold_alarm(parts: np.ndarray, decay: float = 0.95) -> np.ndarray:
        """Peak-hold the DN alarm (~1 s at one sample per 20 ms): the gain controller makes the raw surge flicker."""
        out = parts.copy()
        for k in range(1, len(out)):
            out[k, 2] = max(out[k, 2], out[k - 1, 2] * decay)
        return out

    @staticmethod
    def pain_index(parts: np.ndarray) -> np.ndarray:
        return 100.0 * (parts @ np.array([0.55, 0.25, 0.20]))

    def history(self, first: int, last: int) -> np.ndarray:
        """Samples [first, last) by absolute sample index, clipped to what the ring buffer still holds."""
        first = max(first, self.hist_n - HIST, 0)
        last = min(last, self.hist_n)
        if last <= first:
            return np.zeros((0, len(self.names)), np.float32)
        return self.hist[np.arange(first, last) % HIST].copy()


# --- brain view ----------------------------------------------------------------
VIEW_X, VIEW_Y, VIEW_ZCUT = (0.0, 96000.0), (2000.0, 54000.0), 60000.0   # front view of the brain; +y is ventral
VIEW_SIZES = {"panel": (356, 193), "big": (860, 466)}


class BrainView:
    """Front view of the brain that lights up with the live sim.

    Every neuron is a fiber from its real cell body (MaleCNS soma positions) to the synapse-weighted center of its
    partners' cell bodies, with a tuft of points there for its arbor. Neurons without a soma in the data (most sensory
    neurons) are just the tuft. Real neuron shapes aren't in the pack, so fibers are estimates. Color is the fiber's
    direction (red left-right, green up-down, blue front-back); brightness is how far a neuron fires above its own
    calm rate, on top of a dim image of the whole brain. Neurons in the nerve cord are cut off at the neck.
    """

    FIBER, ARBOR = 18, 3

    def __init__(self, soma: np.ndarray, W, seed: int = 1):
        rng = np.random.default_rng(seed)
        n = len(soma)
        has = ~np.isnan(soma[:, 0])
        S = np.nan_to_num(soma).astype(np.float32)
        M = abs(W).astype(np.float32)
        M = (M + M.T).tocsr()
        tot = M @ has.astype(np.float32)
        ok = tot > 0
        A = (M @ (S * has[:, None])) / np.maximum(tot, 1e-6)[:, None]
        start = np.where(has[:, None], S, A)
        d = A - start
        length = np.linalg.norm(d, axis=1)
        rand = np.abs(rng.normal(size=(n, 3))).astype(np.float32)
        dirv = np.where((length > 1)[:, None], np.abs(d) / np.maximum(length, 1)[:, None], rand / np.linalg.norm(rand, axis=1)[:, None])
        col = dirv ** 2
        col = col / col.max(axis=1, keepdims=True)                   # saturated direction color
        self.col = (0.06 + 0.94 * col).astype(np.float32)

        t = np.linspace(0, 1, self.FIBER, dtype=np.float32)[None, :, None]
        bend = rng.normal(size=(n, 3)).astype(np.float32) * (0.12 * length)[:, None]
        mid = (start + A) / 2 + bend
        fiber = (1 - t) ** 2 * start[:, None] + 2 * (1 - t) * t * mid[:, None] + t ** 2 * A[:, None]
        r = np.clip(0.05 * length, 400, 1600)
        arbor = A[:, None] + rng.normal(size=(n, self.ARBOR, 3)).astype(np.float32) * r[:, None, None]
        pts = np.concatenate([fiber, arbor], 1)                      # (n, samples, 3)
        wts = np.concatenate([np.ones(self.FIBER), np.full(self.ARBOR, 2.0)]).astype(np.float32)
        nid = np.broadcast_to(np.arange(n)[:, None], pts.shape[:2])
        keep = ok[:, None] & (pts[..., 2] < VIEW_ZCUT)

        self.M, self.base, self.gain = {}, {}, {}
        for key, (w, h) in VIEW_SIZES.items():
            px = ((pts[..., 0] - VIEW_X[0]) / (VIEW_X[1] - VIEW_X[0]) * w).astype(np.int32)
            py = ((pts[..., 1] - VIEW_Y[0]) / (VIEW_Y[1] - VIEW_Y[0]) * h).astype(np.int32)
            m = keep & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            P = sp.coo_array((np.broadcast_to(wts, pts.shape[:2])[m], ((py * w + px)[m], nid[m])), shape=(w * h, n)).tocsr()
            self.M[key] = P
            struct = P @ self.col
            p99 = float(np.percentile(struct.max(1), 99.0)) or 1.0
            self.base[key] = struct * (0.5 / p99)
            self.gain[key] = 3.2 / p99
        self.calm = np.full(n, 0.025, np.float32)                   # per-neuron calm rate, spikes per step

    def render(self, key: str, rates: np.ndarray, learn: bool) -> pygame.Surface:
        if learn:
            self.calm += (rates - self.calm) * 0.01
        r = rates / 0.025                                            # 1.0 = 5 Hz
        excess = np.maximum(r - self.calm / 0.025 - 0.3, 0)
        act = (0.04 * r + 2.2 * excess).astype(np.float32)
        light = (self.M[key] @ (act[:, None] * self.col)) * self.gain[key]
        w, h = VIEW_SIZES[key]
        img = (255 * (1 - np.exp(-(self.base[key] + light)))).astype(np.uint8)
        surf = pygame.image.frombuffer(img.tobytes(), (w, h), "RGB").copy()
        # bloom from the firing only, so the wiring stays sharp and activity glows
        hot = (255 * (1 - np.exp(-0.6 * light))).astype(np.uint8)
        glow = pygame.image.frombuffer(hot.tobytes(), (w, h), "RGB")
        glow = pygame.transform.smoothscale(pygame.transform.smoothscale(glow, (w // 6, h // 6)), (w, h))
        surf.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        return surf


# --- fly ragdoll ---------------------------------------------------------------
HEAD, THX, ABD = 0, 1, 2
KNEE = (3, 5, 7, 9, 11, 13)       # near front, mid, hind, then far front, mid, hind
FOOT = (4, 6, 8, 10, 12, 14)
WING = (15, 16)                   # near, far
N_P = 17
STAND = 64                        # thorax height above the floor when standing

REST = np.zeros((N_P, 2))         # facing right, thorax at origin, +y down
REST[HEAD], REST[ABD] = (42, -6), (-50, 6)
for i, (k, f) in enumerate((((44, 22), (58, 60)), ((8, 28), (10, 60)), ((-30, 24), (-44, 60)))):
    REST[KNEE[i]], REST[FOOT[i]] = k, f
    REST[KNEE[i + 3]], REST[FOOT[i + 3]] = (k[0] + 8, k[1] - 4), (f[0] + 10, f[1] - 4)
REST[WING[0]], REST[WING[1]] = (-62, -22), (-54, -30)
RADIUS = np.array([16, 19, 24] + [4] * 12 + [3, 3], float)
PULL = np.array([0.10, 0.10, 0.10] + [0.12, 0.22] * 6 + [0.15, 0.15])
TRIPOD = (0, 1, 0, 1, 0, 1)       # near front/hind + far mid step together

# (a, b, stiffness, shape): shape links only hold legs and wings in place while the fly is a limp ragdoll. They form
# triangles that can't un-flip, so while standing the pose pull shapes the fly instead and they are skipped.
LINKS = [(HEAD, THX, 1.0, False), (THX, ABD, 1.0, False), (HEAD, ABD, 1.0, False)]
for i in range(6):
    LINKS += [(THX, KNEE[i], 0.9, False), (KNEE[i], FOOT[i], 0.9, False), (HEAD if i % 3 == 0 else ABD, KNEE[i], 0.25, True)]
for wtip in WING:
    LINKS += [(THX, wtip, 0.9, False), (ABD, wtip, 0.6, True), (HEAD, wtip, 0.3, True)]
LINK_LEN = [float(np.hypot(*(REST[a] - REST[b]))) for a, b, _, _ in LINKS]
MAX_HEALTH = 100.0


def particle_region(i: int) -> tuple[str, str | None]:
    if i == HEAD:
        return "head", None
    if i in (THX, ABD):
        return "body", None
    if i in WING:
        return "wing", "L" if i == WING[0] else "R"
    return "legs", "L" if (i - 3) // 2 < 3 else "R"


class Fly:
    def __init__(self, x: float):
        self.facing = 1
        self.p = REST + (x, FLOOR - STAND)
        self.prev = self.p.copy()
        self.anchor_x = x
        self.phase = 0.0
        self.grabbed: int | None = None
        self.stun_until = 0.0
        self.escape_until = 0.0
        self.walk_until = 0.0
        self.back_until = 0.0
        self.flail_until = 0.0
        self.run = False
        self.turn_ready = 0.0
        self.escape_ready = 0.0
        self.recover = 1.0
        self.hurt = 0.0
        self.last_hit_x = x
        self.action = "idle"
        self.health = MAX_HEALTH
        self.dead_at: float | None = None
        self.char = 0.0                   # scorch from the blowtorch
        self.burn_until = 0.0
        self.soak = 0.0                   # brake cleaner on the body, 0..1
        self.melt = 0.0                   # how dissolved it is; 1 = puddle
        self.dissolved_at: float | None = None

    @property
    def dead(self) -> bool:
        return self.dead_at is not None

    def nearest(self, pos, max_d: float) -> int | None:
        d = np.hypot(*(self.p - pos).T) - RADIUS
        i = int(np.argmin(d))
        return i if d[i] < max_d else None

    def impulse(self, i: int, v) -> None:
        self.prev[i] -= v

    def stun(self, now: float, s: float) -> None:
        self.stun_until = max(self.stun_until, now + s)
        self.hurt = 1.0

    def escape(self, now: float) -> None:
        away = 1.0 if self.p[THX, 0] >= self.last_hit_x else -1.0
        v = np.array([away * 8.0, -15.0])
        self.prev[:] = self.p - v
        self.escape_until = now + 0.55
        self.escape_ready = now + 1.6
        self.facing = int(away)

    def step(self, now: float, mouse) -> list[tuple[int, float]]:
        """One 60 Hz physics frame. Returns (particle, impact speed) for hard contacts with the arena."""
        escaping = now < self.escape_until
        if self.grabbed is not None or escaping or now < self.stun_until or self.dead:
            self.recover = 0.0
        else:
            self.recover = min(1.0, self.recover + 1 / 30)
        height = (FLOOR - STAND) - self.p[THX, 1]
        strength = self.recover * float(np.clip(1 - height / 160, 0, 1)) * (1 - self.melt) ** 0.5
        if self.grabbed is not None:
            strength = 0.0
        shrink = 1 - 0.35 * self.melt                        # dissolving: the body shrinks and slumps

        v = (self.p - self.prev) * (0.992 - 0.12 * strength)  # standing legs also damp the body
        speed = np.hypot(*v.T)
        v *= np.minimum(1.0, 60.0 / np.maximum(speed, 1e-6))[:, None]
        self.prev = self.p.copy()
        self.p += v
        self.p[:, 1] += (0.45 if escaping else 0.9) * (1 - strength)   # standing: the legs carry the weight
        if escaping:                                         # buzzing glide
            self.p[:, 0] += 0.25 * self.facing

        if strength > 0:
            self._pose(now, strength)
        else:
            self.anchor_x = float(self.p[THX, 0])
        if self.dead:                                        # legs slowly curl in toward the body
            idx = list(KNEE + FOOT)
            self.p[idx] += (self.p[THX] - self.p[idx]) * 0.02
        elif now < self.flail_until or (self.grabbed is not None and random.random() < 0.5):
            idx = list(KNEE + FOOT)
            self.p[idx] += np.random.normal(0, 2.5, (12, 2))

        posed = strength > 0.5
        for _ in range(6):
            if self.grabbed is not None:
                self.p[self.grabbed] = mouse
            for (a, b, stiff, shape), rest in zip(LINKS, LINK_LEN):
                if shape and posed:
                    continue
                d = self.p[b] - self.p[a]
                dist = math.hypot(d[0], d[1]) or 1e-6
                corr = d * (0.5 * stiff * (dist - rest * shrink) / dist)
                self.p[a] += corr
                self.p[b] -= corr
            self._clamp()
        if self.grabbed is not None:
            self.p[self.grabbed] = mouse
        self.hurt = max(0.0, self.hurt - 1 / 20)
        return self._contacts()

    def _pose(self, now: float, strength: float) -> None:
        walking = now < self.walk_until
        backing = now < self.back_until
        speed = 5.0 if self.run else 2.3
        vx = -1.7 * self.facing if backing else speed * self.facing if walking else 0.0
        if vx and not 80 < self.anchor_x + vx < PLAY_W - 80:     # walked into a wall: turn around (game rule)
            self.facing = -self.facing
            vx = -vx
        self.anchor_x = float(np.clip(self.anchor_x + vx, 70, PLAY_W - 70))
        self.anchor_x += 0.05 * (self.p[THX, 0] - self.anchor_x)   # don't drag a body that got knocked away
        self.action = "back up" if backing else ("run" if self.run else "walk") if walking else "idle"
        if vx:
            self.phase += 0.12 * abs(vx)
        bob = 1.5 * math.sin(now * 2.2)
        s = 1 - 0.35 * self.melt
        off = REST * s
        off[:, 0] *= self.facing
        tgt = off + (self.anchor_x, FLOOR - STAND * s + bob)
        for leg in range(6):
            ph = self.phase + math.pi * TRIPOD[leg]
            if vx:
                tgt[FOOT[leg], 0] += 14 * math.cos(ph) * np.sign(vx)
                tgt[FOOT[leg], 1] -= 12 * max(0.0, math.sin(ph))
        d = tgt - self.p
        self.p += d * (PULL * strength)[:, None]
        self.prev += d * (PULL * strength * 0.85)[:, None]

    def _clamp(self) -> None:
        r = RADIUS
        np.clip(self.p[:, 0], r, PLAY_W - r, out=self.p[:, 0])
        np.clip(self.p[:, 1], CEIL + r, FLOOR - r, out=self.p[:, 1])

    def _contacts(self) -> list[tuple[int, float]]:
        hits = []
        r = RADIUS
        v = self.p - self.prev
        for i in range(N_P):
            x, y = self.p[i]
            if y >= FLOOR - r[i] - 0.5 and v[i, 1] > 0:
                if v[i, 1] > 9:
                    hits.append((i, v[i, 1]))
                self.prev[i, 1] = y + v[i, 1] * 0.35
                self.prev[i, 0] = x - v[i, 0] * 0.75
            elif y <= CEIL + r[i] + 0.5 and v[i, 1] < 0:
                if -v[i, 1] > 9:
                    hits.append((i, -v[i, 1]))
                self.prev[i, 1] = y + v[i, 1] * 0.35
            if (x <= r[i] + 0.5 and v[i, 0] < 0) or (x >= PLAY_W - r[i] - 0.5 and v[i, 0] > 0):
                if abs(v[i, 0]) > 9:
                    hits.append((i, abs(v[i, 0])))
                self.prev[i, 0] = x + v[i, 0] * 0.4
        return hits


# --- drawing -------------------------------------------------------------------
def ellipse_pts(c, a: float, b: float, ang: float, n: int = 24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = math.cos(ang), math.sin(ang)
    x, y = a * np.cos(t), b * np.sin(t)
    return [(c[0] + x[k] * ca - y[k] * sa, c[1] + x[k] * sa + y[k] * ca) for k in range(n)]


def aapoly(surf, pts, col) -> None:
    ip = [(int(round(x)), int(round(y))) for x, y in pts]
    gfxdraw.filled_polygon(surf, ip, col)
    gfxdraw.aapolygon(surf, ip, col)


def aacircle(surf, c, r: float, col) -> None:
    x, y, r = int(round(c[0])), int(round(c[1])), max(1, int(round(r)))
    gfxdraw.filled_circle(surf, x, y, r, col)
    gfxdraw.aacircle(surf, x, y, r, col)


def thick_line(surf, a, b, w: float, col) -> None:
    d = np.array(b, float) - np.array(a, float)
    L = math.hypot(*d) or 1
    n = np.array([-d[1], d[0]]) / L * w / 2
    aapoly(surf, [a + n, b + n, b - n, a - n], col)
    aacircle(surf, b, w / 2, col)


def shade(col, hurt: float, dead: bool, char: float = 0.0, melt: float = 0.0):
    if char:
        col = tuple(c * (1 - 0.75 * char) + 22 * char for c in col)
    if dead:
        g = sum(col) / 3
        col = tuple(0.45 * c + 0.55 * g * 0.8 for c in col)
    if melt:                                          # dissolving: sickly, glassy, see-through
        col = tuple(c * (1 - 0.6 * melt) + t * 0.6 * melt for c, t in zip(col, (150, 160, 115)))
    rgb = tuple(int(min(255, c + (255 - c) * hurt * 0.7)) for c in col)
    return (*rgb, int(255 * (1 - 0.6 * melt))) if melt else rgb


def draw_puddle(surf: pygame.Surface, fly: Fly, now: float) -> None:
    """What's left after the brake cleaner: a spreading puddle, a wing and two red eye specks."""
    x = float(fly.p[THX, 0])
    e = min(1.0, (now - fly.dissolved_at) / 1.5)
    w = 70 + 60 * e
    gfxdraw.filled_ellipse(surf, int(x), FLOOR - 3, int(w), int(9 + 3 * e), (110, 105, 60, 170))
    gfxdraw.aaellipse(surf, int(x), FLOOR - 3, int(w), int(9 + 3 * e), (160, 150, 90, 200))
    gfxdraw.filled_ellipse(surf, int(x - w * 0.3), FLOOR - 5, int(w * 0.35), 4, (185, 175, 120, 120))
    wing = ellipse_pts((x + 22, FLOOR - 8), 26, 6, 0.15)
    gfxdraw.filled_polygon(surf, [(int(a), int(b)) for a, b in wing], (205, 218, 238, 90))
    for dx in (-8, 5):
        aacircle(surf, (x + dx, FLOOR - 6), 2.5, (180, 30, 26))
    for k in range(3):                               # slow bubbles popping on the surface
        ph = (now * 0.8 + k / 3) % 1.0
        bx = x + (k - 1) * w * 0.45
        gfxdraw.aacircle(surf, int(bx), int(FLOOR - 6 - 10 * ph), int(2 + 3 * ph), (200, 200, 160, int(200 * (1 - ph))))


def draw_fly(surf: pygame.Surface, fly: Fly, now: float) -> None:
    if fly.dissolved_at is not None:
        draw_puddle(surf, fly, now)
        return
    p = fly.p
    axis = p[HEAD] - p[ABD]
    ang = math.atan2(axis[1], axis[0])
    fwd = axis / (np.hypot(*axis) or 1)
    up = np.array([fwd[1], -fwd[0]])
    if fwd[0] < 0:                                   # keep "up" above the body when it faces left
        up = -up
    hurt, dead = fly.hurt, fly.dead
    flap = not dead and (now < fly.escape_until or (fly.grabbed is not None and random.random() < 0.3))
    k_ = 1 - 0.4 * fly.melt                           # parts shrink as it dissolves

    def c(col):
        return shade(col, hurt, dead, fly.char, fly.melt)

    def hip(k):
        return p[THX] + fwd * (14 - 12 * (k % 3)) + up * -10

    def leg(k, col, w):
        h, kn, ft = hip(k), p[KNEE[k]], p[FOOT[k]]
        thick_line(surf, h, kn, w, col)
        thick_line(surf, kn, ft, w * 0.65, col)

    def wing(i, alpha):
        base = p[THX] + up * 12
        tip = p[WING[i]]
        if flap:
            tip = base + (tip - base) * 0.8 + up * (38 * math.sin(now * 90 + i))
        mid = (base + tip) / 2
        span = np.hypot(*(tip - base))
        pts = ellipse_pts(mid, span / 2 + 6, 13, math.atan2(*(tip - base)[::-1]))
        ip = [(int(x), int(y)) for x, y in pts]
        gfxdraw.filled_polygon(surf, ip, (205, 218, 238, alpha))
        gfxdraw.aapolygon(surf, ip, (235, 240, 252, min(255, alpha + 90)))
        for off in (-4, 4):
            q = tip + up * off
            gfxdraw.line(surf, int(base[0]), int(base[1]), int(q[0]), int(q[1]), (170, 180, 200, alpha + 30))

    wing(1, 55)
    for k in (3, 4, 5):
        leg(k, c((58, 38, 18)), 4.5)
    ac = p[ABD]
    aang = math.atan2(*(p[THX] - p[ABD])[::-1])
    along = np.array([math.cos(aang), math.sin(aang)])
    aapoly(surf, ellipse_pts(ac, 30 * k_, 21 * k_, aang), c((176, 116, 44)))
    for s in (-15, -4, 7):
        aapoly(surf, ellipse_pts(ac + along * s * k_, 4 * k_, 19 * k_, aang, 14), c((82, 52, 24)))
    aapoly(surf, ellipse_pts(ac + (up * 7 + along * 4) * k_, 16 * k_, 6 * k_, aang, 16), c((214, 158, 80)))   # sheen
    aapoly(surf, ellipse_pts(p[THX], 23 * k_, 19 * k_, ang), c((156, 102, 40)))
    aapoly(surf, ellipse_pts(p[THX] + up * 8 * k_, 13 * k_, 6 * k_, ang, 16), c((196, 140, 70)))
    for k in range(5):                                # bristles
        b0 = p[THX] + up * 17 + fwd * (k * 6 - 12)
        b1 = b0 + up * 7 - fwd * 3
        gfxdraw.line(surf, int(b0[0]), int(b0[1]), int(b1[0]), int(b1[1]), c((60, 40, 18)))
    hc = p[HEAD]
    aacircle(surf, hc, 15 * k_, c((146, 96, 38)))
    eye = hc + (fwd * 2 + up * 2) * k_
    aapoly(surf, ellipse_pts(eye, 11 * k_, 12 * k_, ang), c((196, 30, 26)))
    if dead:
        for sgn in (-1, 1):
            a0 = eye + np.array([-6, -6 * sgn])
            a1 = eye + np.array([6, 6 * sgn])
            thick_line(surf, a0, a1, 3, (30, 20, 20))
    else:
        for fx, fy in ((-4, 3), (0, 5), (4, 2), (-2, -1), (3, -3)):
            gfxdraw.pixel(surf, int(eye[0] + fx), int(eye[1] + fy), (120, 16, 14))
        aacircle(surf, eye + (-3, -4), 3, (255, 170, 160))
    for da in (0.0, 0.3):
        a0 = hc + fwd * 12 + up * 8
        a1 = a0 + fwd * 10 + up * (10 - 7 * da)
        thick_line(surf, a0, a1, 2, c((70, 45, 20)))
    thick_line(surf, hc + fwd * 8 - up * 10, hc + fwd * 14 - up * 20, 3, c((70, 45, 20)))
    for k in (0, 1, 2):
        leg(k, c((86, 58, 26)), 5.5)
    wing(0, 75)

    if fly.soak > 0.05:                              # foam where the solvent is eating in
        rng = random.Random(int(now * 8))
        for _ in range(int(4 + 10 * fly.soak)):
            i = rng.choice((HEAD, THX, ABD, ABD))
            fx, fy = p[i] + (rng.uniform(-22, 22) * k_, rng.uniform(-20, 16) * k_)
            gfxdraw.aacircle(surf, int(fx), int(fy), rng.randint(2, 5), (235, 240, 220, int(90 + 120 * fly.soak)))
    if dead:
        e = min(1.0, (now - fly.dead_at) / 2.0)
        halo = hc + np.array([0, -34 - 10 * e])
        gfxdraw.aaellipse(surf, int(halo[0]), int(halo[1]), 16, 5, (255, 225, 120))
        gfxdraw.aaellipse(surf, int(halo[0]), int(halo[1]), 15, 4, (255, 225, 120))
    elif now < fly.stun_until:                       # dizzy stars
        for k in range(4):
            a = now * 6 + k * math.pi / 2
            sx, sy = hc[0] + 28 * math.cos(a), hc[1] - 30 + 7 * math.sin(a)
            pts = [(sx + (7 if j % 2 == 0 else 3) * math.cos(j * math.pi / 5 - math.pi / 2),
                    sy + (7 if j % 2 == 0 else 3) * math.sin(j * math.pi / 5 - math.pi / 2)) for j in range(10)]
            aapoly(surf, pts, (255, 215, 60))


def draw_swatter(surf, pos, ph: float) -> None:
    """ph 0..1: 0-0.3 swing down, 0.3-0.55 pressed, 0.55-1 lift and fade."""
    if ph < 0.3:
        angle, alpha = -0.9 * (1 - (ph / 0.3) ** 2), 255
    elif ph < 0.55:
        angle, alpha = 0.0, 255
    else:
        e = (ph - 0.55) / 0.45
        angle, alpha = -1.1 * e, int(255 * (1 - e))
    pivot = np.array([pos[0] + 260, pos[1] + 330])
    d = np.array(pos, float) - pivot
    L = float(np.hypot(*d))
    base = math.atan2(d[1], d[0]) + angle
    u = np.array([math.cos(base), math.sin(base)])
    v = np.array([-u[1], u[0]])
    head = pivot + u * L
    hw, hl = 70, 80
    corners = [head + u * hl * a + v * hw * b for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    thick_line(surf, pivot, head - u * hl, 12, (120, 78, 40, alpha))
    aapoly(surf, corners, (210, 40, 45, min(alpha, 220)))
    for i in np.linspace(-0.8, 0.8, 7):
        for a0, a1 in ((head - u * hl + v * hw * i, head + u * hl + v * hw * i), (head + u * hl * i - v * hw, head + u * hl * i + v * hw)):
            gfxdraw.line(surf, int(a0[0]), int(a0[1]), int(a1[0]), int(a1[1]), (250, 130, 130, min(alpha, 150)))
    gfxdraw.aapolygon(surf, [(int(x), int(y)) for x, y in corners], (110, 18, 22, alpha))


def draw_icon(surf, name: str, c, col) -> None:
    x, y = c
    if name == "hand":
        aacircle(surf, (x, y + 4), 9, col)
        for k in range(4):
            thick_line(surf, (x - 7 + k * 4.5, y), (x - 8 + k * 5, y - 11 + abs(k - 1.5) * 2), 4, col)
    elif name == "flick":
        aacircle(surf, (x - 4, y + 3), 7, col)
        thick_line(surf, (x - 2, y - 1), (x + 10, y - 9), 4, col)
        for k in (-1, 0, 1):
            thick_line(surf, (x + 13, y - 12 + k * 6), (x + 18, y - 14 + k * 8), 2, col)
    elif name == "swatter":
        aapoly(surf, [(x - 3, y - 13), (x + 11, y - 13), (x + 11, y + 1), (x - 3, y + 1)], col)
        thick_line(surf, (x + 2, y + 1), (x - 8, y + 12), 3, col)
    elif name == "cleaner":
        aapoly(surf, [(x - 8, y - 6), (x + 2, y - 6), (x + 2, y + 13), (x - 8, y + 13)], col)
        aapoly(surf, [(x - 6, y - 11), (x, y - 11), (x, y - 6), (x - 6, y - 6)], col)
        for k in range(3):
            aacircle(surf, (x + 8 + k * 4, y - 12 + (k - 1) * 4), 2, (160, 220, 255))
    elif name == "torch":
        aapoly(surf, [(x - 12, y + 2), (x + 2, y - 4), (x + 5, y + 2), (x - 9, y + 8)], col)
        aapoly(surf, [(x + 4, y - 4), (x + 16, y - 14), (x + 11, y - 1)], (255, 140, 40))
        aapoly(surf, [(x + 5, y - 3), (x + 12, y - 9), (x + 9, y - 2)], (255, 230, 120))
    else:
        aacircle(surf, (x - 2, y + 3), 10, col)
        thick_line(surf, (x + 5, y - 5), (x + 10, y - 11), 3, col)
        aacircle(surf, (x + 11, y - 13), 3, AMBER)


# --- game ----------------------------------------------------------------------
TOOLS = (("hand", "HAND", "drag + throw"), ("flick", "FLICK", "click"), ("swatter", "SWATTER", "click"),
         ("bomb", "BOMB", "click to drop"), ("torch", "TORCH", "hold: max pain"), ("cleaner", "CLEANER", "hold: melt it"))
TORCH_KEYS = (("head", None), ("body", None), ("legs", "L"), ("legs", "R"), ("wing", "L"), ("wing", "R"), ("heat", None))
OUCH = ("BONK!", "OOF!", "SPLAT!", "THWACK!", "BZZT!", "OW!")
AUTOPSY_DELAY = 3.0          # seconds of flatline shown before the report


def make_background() -> pygame.Surface:
    y = np.linspace(0, 1, H)[None, :, None]
    x = np.linspace(-1, 1, PLAY_W)[:, None, None]
    top, bottom = np.array([28, 34, 48]), np.array([14, 16, 22])
    img = top * (1 - y) + bottom * y
    yy = np.linspace(-1, 1, H)[None, :, None]
    spot = np.exp(-(x ** 2 * 1.6 + (yy - 0.35) ** 2 * 2.2))    # soft light pooled over the floor
    img = img + spot * np.array([16, 18, 22])
    floor_y = np.arange(H)[None, :, None]
    fl = floor_y >= FLOOR
    wood = np.array([46, 38, 32]) - (floor_y - FLOOR) * 0.12
    img = np.where(fl, wood + spot * 20, img)
    surf = pygame.surfarray.make_surface(np.clip(img, 0, 255).astype(np.uint8))
    for k in range(-12, 14):                         # floor boards in perspective
        x0 = PLAY_W / 2 + k * 70
        pygame.draw.aaline(surf, (38, 31, 26), (x0, FLOOR), (PLAY_W / 2 + k * 150, H))
    for yy_ in (FLOOR + 22, FLOOR + 58, FLOOR + 104):
        pygame.draw.line(surf, (40, 33, 28), (0, yy_), (PLAY_W, yy_))
    pygame.draw.line(surf, (92, 78, 64), (0, FLOOR), (PLAY_W, FLOOR), 2)
    return surf


def make_shadow() -> pygame.Surface:
    w, h = 220, 40
    x = np.linspace(-1, 1, w)[:, None]
    y = np.linspace(-1, 1, h)[None, :]
    a = np.clip(1 - (x ** 2 + y ** 2), 0, 1) ** 1.6 * 150
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels_alpha(s)[:] = a.astype(np.uint8)
    return s


class Game:
    def __init__(self, screen, brain: Brain, view: BrainView):
        self.screen, self.brain = screen, brain
        self.tool = 0
        self.bucks = 0
        self.kills = 0
        self.f_small = pygame.font.SysFont("consolas", 13)
        self.f_text = pygame.font.SysFont("segoeui,consolas", 15)
        self.f_bold = pygame.font.SysFont("segoeuisemibold,segoeui,consolas", 16, bold=True)
        self.f_head = pygame.font.SysFont("segoeuiblack,segoeui,consolas", 20, bold=True)
        self.f_title = pygame.font.SysFont("segoeuiblack,segoeui,consolas", 34, bold=True)
        self.f_big = pygame.font.Font(pygame.font.match_font("impact,arialblack,arial"), 40)
        self.bg = make_background()
        self.shadow = make_shadow()
        self.view = view
        self.view_surf: dict[str, pygame.Surface] = {}
        self.view_rect = pygame.Rect(0, 0, 0, 0)
        self.big_view = False
        self.view_stop = False
        view.calm[:] = brain.sim.activity.rates()   # the warmed-up brain's own resting rates
        threading.Thread(target=self._view_loop, name="brain-view", daemon=True).start()
        self.new_fly()

    def new_fly(self) -> None:
        self.fly = Fly(PLAY_W / 2)
        self.hits = 0
        self.popups: list[list] = []
        self.bombs: list[dict] = []
        self.flashes: list[list] = []
        self.swats: list[list] = []
        self.dust: list[list] = []
        self.shake_until = 0.0
        self.pending_hits: dict[tuple[str, str | None], float] = {}
        self.pending_damage = 0.0
        self.damage_src = ""
        self.killed_by = ""
        self.log: list[tuple[float, str]] = []
        self.born = time.perf_counter()
        self.report: dict | None = None
        self.torching = False
        self.flames: list[list] = []
        self.mist: list[list] = []
        self.brain.sedation = 0.0
        self.pain = 0.0
        self.pain_parts = np.zeros(3)
        self.pain_peak = 0.0
        self.pain_max_s = 0.0
        self.pain_trace: list[float] = []
        if self.brain.dead:
            self.brain.revive()

    def _view_loop(self) -> None:
        """Renders the brain view at ~20 Hz on its own thread (10-25 ms per render; the sparse math releases the GIL)."""
        while not self.view_stop:
            t0 = time.perf_counter()
            key = "big" if self.big_view else "panel"
            br = self.brain
            calm = not br.dead and br.sedation == 0 and br.steps - br.last_poke > CALM_STEPS
            self.view_surf[key] = self.view.render(key, br.sim.activity.rates(), learn=calm)
            time.sleep(max(0.005, 0.05 - (time.perf_counter() - t0)))

    def _view_surface(self, key: str) -> pygame.Surface:
        surf = self.view_surf.get(key)
        if surf is None:
            surf = pygame.Surface(VIEW_SIZES[key])
            surf.fill((4, 5, 8))
        return surf

    def _draw_big_view(self, surf) -> None:
        w, h = VIEW_SIZES["big"]
        veil = pygame.Surface((PLAY_W, h + 96), pygame.SRCALPHA)
        pygame.draw.rect(veil, (4, 5, 8, 235), veil.get_rect(), border_radius=14)
        surf.blit(veil, (0, 8))
        surf.blit(self._view_surface("big"), ((PLAY_W - w) // 2, 58))
        self._text(surf, "THE FLY'S BRAIN, LIVE", (22, 18), INK, self.f_head)
        self._text(surf, "front view   |   color = fiber direction: red left-right, green up-down, blue front-back   |   "
                         "bright = firing above normal", (24, 42), LABEL, self.f_small)
        self._text(surf, "B to close", (PLAY_W - 22, 20), LABEL, self.f_small, "topright")
        self._text(surf, "Fibers run from each neuron's real cell body toward its synaptic partners (estimated shapes).",
                   (24, 58 + h + 14), DIM, self.f_small)

    def hit(self, i: int, strength: float) -> None:
        key = particle_region(i)
        self.pending_hits[key] = max(self.pending_hits.get(key, 0.0), strength)

    def damage(self, amount: float, source: str) -> None:
        if amount > self.pending_damage:
            self.damage_src = source
        self.pending_damage += amount

    def popup(self, pos, text: str, color=(255, 245, 235), force=False) -> None:
        now = time.perf_counter()
        if self.popups and now - self.popups[-1][3] < 0.3 and not force:
            return
        x = float(np.clip(pos[0], 90, PLAY_W - 90))
        y = float(np.clip(pos[1], CEIL + 60, FLOOR - 20))
        self.popups.append([x, y, text, now, color])

    def puff(self, pos, n: int, spread: float = 3.0) -> None:
        for _ in range(n):
            self.dust.append([pos[0], pos[1], random.uniform(-spread, spread), random.uniform(-spread, 0.3),
                              time.perf_counter(), random.uniform(0.35, 0.8), random.uniform(3, 7)])

    def note(self, text: str) -> None:
        self.log.append((time.perf_counter(), text))
        self.log = self.log[-7:]

    # --- tools ---
    def use_tool(self, pos, now: float) -> None:
        fly = self.fly
        name = TOOLS[self.tool][0]
        if name == "hand":
            i = fly.nearest(pos, 40)
            if i is not None:
                fly.grabbed = i
                fly.last_hit_x = pos[0]
                self.hit(i, 0.25)
        elif name == "flick":
            d = np.hypot(*(fly.p - pos).T) - RADIUS
            near = np.flatnonzero(d < 60)
            if len(near):
                fly.last_hit_x = pos[0]
                for i in near:
                    dirv = fly.p[i] - pos
                    dirv = dirv / (np.hypot(*dirv) or 1) + (0, -0.6)
                    s = 1 - max(d[i], 0) / 60
                    fly.impulse(i, dirv * (10 + 16 * s))
                    self.hit(i, 0.35 + 0.4 * s)
                fly.stun(now, 0.35)
                self.damage(4, "a flick")
                self.popup(pos, "FLICK!")
        elif name == "swatter":
            if not self.swats or now - self.swats[-1][1] > 0.35:
                self.swats.append([pos, now, False])
        elif name == "bomb" and len(self.bombs) < 3:
            self.bombs.append({"p": np.array(pos, float), "v": np.zeros(2), "t": now})
        elif name in ("torch", "cleaner"):
            self.torching = True

    def _spray(self, mouse, now: float) -> None:
        """Brake cleaner mist toward the fly. Soaks the body (it dissolves) and hits the smell and taste neurons."""
        fly = self.fly
        nozzle = np.array(mouse, float)
        to_fly = fly.p[THX] - nozzle
        dist = float(np.hypot(*to_fly))
        aim = to_fly / dist if dist > 1 else np.array([1.0, 0.0])
        self.torch_aim = aim
        for _ in range(5):
            a = math.atan2(aim[1], aim[0]) + random.uniform(-0.3, 0.3)
            sp = random.uniform(6, 10)
            self.mist.append([nozzle[0], nozzle[1], math.cos(a) * sp, math.sin(a) * sp, now, random.uniform(0.35, 0.6)])
        rel = fly.p - nozzle
        d = np.hypot(*rel.T)
        cosang = (rel @ aim) / np.maximum(d, 1e-6)
        if fly.dissolved_at is not None or not np.any((d < 230) & (cosang > 0.8)):
            return
        fly.soak = min(1.0, fly.soak + 0.04)
        fly.last_hit_x = nozzle[0]
        if not fly.dead:
            self.brain.poke("smell", None, 1.0)
            self.brain.poke("taste", None, 0.8)
            self.damage(0.12, "brake cleaner")
            self.bucks += 1
            if int(now * 2) != int((now - 1 / 60) * 2):
                self.hits += 1
            if random.random() < 0.02:
                self.popup(fly.p[HEAD] + (0, -60), random.choice(("FSSSSH!", "MELTING!", "IT BURNS!")), (170, 230, 255))

    def _dissolve(self, now: float) -> None:
        """Soaked flies keep dissolving; the solvent also damps the whole brain (game rule, see docstring)."""
        fly = self.fly
        if fly.dissolved_at is not None:
            return
        if fly.soak > 0.02:
            fly.melt = min(1.0, fly.melt + 0.0045 * fly.soak)
            fly.soak *= 0.997
            if not fly.dead:
                self.damage(0.25 * fly.soak, "brake cleaner")
        if not fly.dead:
            self.brain.sedation = 0.25 * fly.melt ** 2.5          # mild at first, so it can still try to flee
        if fly.melt >= 1.0:
            fly.dissolved_at = now
            self.puff((fly.p[THX, 0], FLOOR - 10), 14, 3)
            self.popup(fly.p[THX] + (0, -60), "DISSOLVED", (170, 230, 255), force=True)
            if not fly.dead:
                self.damage(MAX_HEALTH, "brake cleaner")

    def _torch(self, mouse, now: float) -> None:
        """Flame jet from the cursor toward the fly. Heat reaches the whole body: every touch and heat neuron, full strength."""
        fly = self.fly
        nozzle = np.array(mouse, float)
        to_fly = fly.p[THX] - nozzle
        dist = float(np.hypot(*to_fly))
        aim = to_fly / dist if dist > 1 else np.array([1.0, 0.0])
        self.torch_aim = aim
        for _ in range(7):
            a = math.atan2(aim[1], aim[0]) + random.uniform(-0.22, 0.22)
            sp = random.uniform(8, 13)
            self.flames.append([nozzle[0], nozzle[1], math.cos(a) * sp, math.sin(a) * sp, now, random.uniform(0.22, 0.4)])
        rel = fly.p - nozzle
        d = np.hypot(*rel.T)
        cosang = (rel @ aim) / np.maximum(d, 1e-6)
        inside = np.flatnonzero((d < 200) & (cosang > 0.85))
        if len(inside):
            fly.burn_until = now + 0.9                  # it catches fire, so dodging the jet doesn't cool it
            for i in inside:
                fly.impulse(i, aim * 0.5 + (0, -0.2))
        if now >= fly.burn_until:
            return
        fly.char = min(1.0, fly.char + 0.004)
        fly.hurt = max(fly.hurt, 0.5)
        fly.last_hit_x = nozzle[0]
        if fly.dead:
            return
        for region, side in TORCH_KEYS:
            self.brain.poke(region, side, 1.0)
        self.damage(0.35, "the blowtorch")
        self.bucks += 1
        if int(now * 2) != int((now - 1 / 60) * 2):
            self.hits += 1
        if random.random() < 0.02:
            self.popup(fly.p[HEAD] + (0, -60), random.choice(("SIZZLE!", "TSSSS!", "HOT HOT!")), (255, 150, 60))

    def _swat_impact(self, pos, now: float) -> None:
        fly = self.fly
        d = np.hypot(*(fly.p - pos).T) - RADIUS
        near = np.flatnonzero(d < 95)
        self.shake_until = now + 0.18
        if len(near):
            fly.last_hit_x = pos[0]
            for i in near:
                fly.impulse(i, np.array([(fly.p[i, 0] - pos[0]) * 0.15, 32.0]))
                self.hit(i, 1.0)
            fly.stun(now, 1.8)
            self.damage(14, "the swatter")
            self.popup((pos[0], pos[1] - 90), "SWAT!", (255, 230, 120))
            self.puff((pos[0], min(pos[1] + 40, FLOOR)), 10, 4)

    def _explode(self, b: dict, now: float) -> None:
        fly = self.fly
        pos = b["p"]
        self.flashes.append([pos.copy(), now])
        self.shake_until = now + 0.4
        self.puff(pos, 24, 7)
        d = np.hypot(*(fly.p - pos).T)
        worst = 0.0
        for i in np.flatnonzero(d < 330):
            f = 48 * (1 - d[i] / 330) ** 1.3
            dirv = (fly.p[i] - pos) / (d[i] or 1) + (0, -0.8)
            fly.impulse(i, dirv * f)
            if f > 4:
                self.hit(i, f / 40)
                worst = max(worst, f)
        if worst:
            fly.last_hit_x = pos[0]
            fly.stun(now, 2.4)
            self.damage(32 * worst / 48, "a bomb")
        self.popup((pos[0], pos[1] - 70), "KABOOM!", (255, 160, 60), force=True)

    # --- per frame ---
    def update(self, now: float, mouse) -> None:
        fly, br = self.fly, self.brain
        for i, sp in fly.step(now, np.array(mouse, float)):
            s = float(np.clip((sp - 9) / 35, 0.05, 1))
            self.hit(i, s)
            if sp > 18:
                self.damage(min(8.0, (sp - 18) * 0.35), "the wall" if fly.p[i, 1] < FLOOR - 30 else "the floor")
            if sp > 14 and fly.p[i, 1] > FLOOR - 30:
                self.puff((fly.p[i, 0], FLOOR - 2), 3)
            if sp > 24 and not fly.dead:
                fly.stun(now, 0.8 * s + 0.3)
                if random.random() < 0.5:
                    self.popup(fly.p[i] + (0, -40), random.choice(OUCH))
        for sw in self.swats:
            if not sw[2] and now - sw[1] > 0.1:
                sw[2] = True
                self._swat_impact(sw[0], now)
        self.swats = [s for s in self.swats if now - s[1] < 0.55]
        for b in list(self.bombs):
            b["v"][1] += 0.9
            b["p"] += b["v"]
            if b["p"][1] > FLOOR - 16:
                b["p"][1], b["v"][:] = FLOOR - 16, 0
            if now - b["t"] > 1.5:
                self.bombs.remove(b)
                self._explode(b, now)
        for d in self.dust:
            d[0] += d[2]
            d[1] += d[3]
            d[3] -= 0.05
        self.dust = [d for d in self.dust if now - d[4] < d[5]]
        if self.torching and TOOLS[self.tool][0] == "torch" and self.report is None:
            self._torch(mouse, now)
        if self.torching and TOOLS[self.tool][0] == "cleaner" and self.report is None:
            self._spray(mouse, now)
        self._dissolve(now)
        for fl in self.flames:
            fl[0] += fl[2]
            fl[1] += fl[3]
            fl[2] *= 0.96
            fl[3] = fl[3] * 0.96 - 0.25
        self.flames = [fl for fl in self.flames if now - fl[4] < fl[5]]
        for m in self.mist:
            m[0] += m[2]
            m[1] += m[3]
            m[2] *= 0.94
            m[3] = m[3] * 0.94 + 0.05
        self.mist = [m for m in self.mist if now - m[4] < m[5]]

        parts = br.pain_parts(br.fast, br.base)[0]
        parts[2] = max(parts[2], self.pain_parts[2] * 0.96)
        self.pain_parts += (parts - self.pain_parts) * 0.15
        self.pain += (float(br.pain_index(parts)) - self.pain) * 0.15
        if not fly.dead:
            self.pain_peak = max(self.pain_peak, self.pain)
            if self.pain >= 99:
                self.pain_max_s += 1 / 60
        self.pain_trace.append(self.pain)
        self.pain_trace = self.pain_trace[-720:]

        if not fly.dead:
            for (region, side), s in self.pending_hits.items():
                br.poke(region, side, s)
                self.bucks += int(1 + 9 * s)
                self.hits += 1
            if self.pending_damage:
                fly.health = max(0.0, fly.health - self.pending_damage)
                if fly.health <= 0:
                    self._die(now)
        self.pending_hits.clear()
        self.pending_damage = 0.0
        if fly.dead:
            if self.report is None and now - fly.dead_at > AUTOPSY_DELAY:
                self.report = self._autopsy(now)
            return

        # reactions read from the descending neurons
        free = fly.grabbed is None and now >= fly.escape_until
        lv = {n: br.level(n) for n in ("jump", "run", "kick", "walk", "back", "turn_l", "turn_r")}
        away = 1 if fly.p[THX, 0] >= fly.last_hit_x else -1
        if lv["jump"] > THRESH["jump"] and now >= fly.escape_ready and free:
            fly.stun_until = 0.0                             # the reflex beats the dizziness
            fly.escape(now)
            self.note(f"JUMP     head-touch DNs x{lv['jump']:.1f}")
            self.popup(fly.p[HEAD] + (0, -60), "YIKES!", (160, 230, 255))
        if lv["run"] > THRESH["run"] and now >= fly.walk_until and free:
            fly.stun_until = min(fly.stun_until, now + 0.2)
            fly.facing, fly.walk_until, fly.back_until, fly.run = away, now + 1.1, 0.0, True
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
        turn = lv["turn_r"] - lv["turn_l"]
        if abs(turn) > THRESH["turn"] and now >= fly.turn_ready and free and now >= fly.walk_until:
            new = 1 if turn > 0 else -1
            fly.turn_ready = now + 1.5
            if new != fly.facing:
                fly.facing = new
                self.note(f"TURN {'R' if new > 0 else 'L'}   DNa01/02 R-L {turn:+.1f}")

    def _die(self, now: float) -> None:
        fly = self.fly
        fly.dead_at = now
        fly.grabbed = None
        self.killed_by = self.damage_src or "being kicked"
        self.kills += 1
        self.brain.kill()
        self.note("DIED     brain drive cut, activity fading")
        self.popup(fly.p[HEAD] + (0, -70), "K.O.!", (255, 90, 80), force=True)
        self.shake_until = now + 0.3

    def _autopsy(self, now: float) -> dict:
        br = self.brain
        ds = br.death_sample
        base = br.death_base
        last = br.history(ds - 100, ds)                  # last 2 s alive
        after = br.history(ds + 50, ds + 150)            # 1-3 s after death
        rows = []
        names = ["whole brain"] + [n for n, _ in POPS] + ["head", "body", "legs", "wing", "heat", "smell", "taste", "jump", "run", "kick", "walk", "back"]
        pretty = {"head": "touch: head", "body": "touch: body", "legs": "touch: legs", "wing": "touch: wings", "heat": "heat sensors",
                  "smell": "smell ORNs", "taste": "taste neurons",
                  "jump": "DNs: jump group", "run": "DNs: run group", "kick": "DNs: kick group", "walk": "DNs: DNp09 walk",
                  "back": "DNs: MDN back up"}
        for n in names:
            i = br.col[n]
            b = float(base[i])
            f = float(last[:, i].mean()) if len(last) else 0.0
            a = float(after[:, i].mean()) if len(after) else 0.0
            rows.append({"name": pretty.get(n, n), "base": b, "final": f, "after": a,
                         "ratio": (f + 0.05) / (b + 0.05), "section": "pop" if n in dict(POPS) or n == "whole brain" else "detail"})
        tl = br.history(ds - 1000, ds + 150)             # 20 s before death to 3 s after
        start = max(ds - 1000, br.hist_n - HIST, 0)
        touch_idx = [br.col[n] for n in ("head", "body", "legs", "wing")]
        tsz = br.g_size[touch_idx]
        series = []
        for label, vals, b in (
            ("whole brain", tl[:, br.col["whole brain"]], base[br.col["whole brain"]]),
            ("touch neurons", (tl[:, touch_idx] * tsz).sum(1) / tsz.sum(), float((base[touch_idx] * tsz).sum() / tsz.sum())),
            ("descending", tl[:, br.col["descending"]], base[br.col["descending"]]),
        ):
            series.append((label, vals / max(float(b), 1e-3)))
        top = max(rows[1:], key=lambda r: r["ratio"])
        pain = br.pain_index(br.hold_alarm(br.pain_parts(tl, base))) if len(tl) else np.zeros(0)
        last_pain = pain[max(0, ds - start - 100):ds - start].mean() if len(pain) else 0.0
        return {"rows": rows, "series": series, "death_at": ds - start, "n": len(tl), "top": top, "pain": pain,
                "pain_last": float(last_pain), "pain_peak": self.pain_peak, "pain_max_s": self.pain_max_s,
                "alive_s": self.fly.dead_at - self.born, "hits": self.hits, "by": self.killed_by}

    # --- render ---
    def draw(self, now: float, mouse) -> None:
        scr = self.screen
        scr.fill(BG)
        arena = self.bg.copy()
        fly = self.fly
        lift = float(np.clip((FLOOR - fly.p[:, 1].max()) / 300, 0, 1))
        sh = pygame.transform.smoothscale(self.shadow, (int(200 * (1 - 0.5 * lift)), int(26 * (1 - 0.5 * lift))))
        sh.set_alpha(int(255 * (1 - 0.7 * lift)))
        arena.blit(sh, sh.get_rect(center=(int(fly.p[THX, 0]), FLOOR + 2)))

        for b in self.bombs:
            bx, by = b["p"]
            aacircle(arena, (bx, by), 16, (34, 34, 40))
            aacircle(arena, (bx - 5, by - 5), 4, (100, 100, 112))
            thick_line(arena, (bx + 8, by - 12), (bx + 14, by - 22), 3, (180, 150, 90))
            if int(now * 12) % 2:
                aacircle(arena, (bx + 15, by - 24), 5, (255, 220, 80))
        draw_fly(arena, fly, now)
        for sw in self.swats:
            draw_swatter(arena, sw[0], (now - sw[1]) / 0.55)
        for d in self.dust:
            e = (now - d[4]) / d[5]
            aacircle(arena, (d[0], d[1]), d[6] * (0.6 + e), (170, 160, 150, int(120 * (1 - e))))
        for fl in self.flames:
            e = (now - fl[4]) / fl[5]
            col = (255, 240, 170) if e < 0.25 else (255, 160, 40) if e < 0.55 else (220, 60, 30) if e < 0.8 else (90, 80, 80)
            aacircle(arena, (fl[0], fl[1]), 4 + 16 * e, (*col, int(210 * (1 - e ** 2))))
        if now < fly.burn_until:                         # small flames licking off the body
            for i in (HEAD, THX, ABD):
                fx, fy = fly.p[i] + (random.uniform(-14, 14), random.uniform(-18, -4))
                aacircle(arena, (fx, fy), random.uniform(4, 9), (255, random.randint(110, 200), 40, 170))
        for m in self.mist:
            e = (now - m[4]) / m[5]
            aacircle(arena, (m[0], m[1]), 5 + 22 * e, (200, 230, 250, int(90 * (1 - e))))
        if self.torching and TOOLS[self.tool][0] == "cleaner" and self.report is None and mouse[0] < PLAY_W:
            aim = getattr(self, "torch_aim", np.array([1.0, 0.0]))
            m = np.array(mouse, float)
            thick_line(arena, m - aim * 78, m - aim * 12, 26, (200, 40, 40))           # the can
            thick_line(arena, m - aim * 60, m - aim * 40, 27, (235, 235, 240))         # label band
            thick_line(arena, m - aim * 12, m - aim * 2, 8, (60, 60, 66))              # nozzle
        if self.torching and TOOLS[self.tool][0] == "torch" and self.report is None and mouse[0] < PLAY_W:
            aim = getattr(self, "torch_aim", np.array([1.0, 0.0]))
            m = np.array(mouse, float)
            thick_line(arena, m - aim * 70, m - aim * 8, 16, (110, 116, 128))
            thick_line(arena, m - aim * 70, m - aim * 50, 18, (200, 60, 50))
            aacircle(arena, m, 5, (255, 250, 220))
        for fl in list(self.flashes):
            e = (now - fl[1]) / 0.35
            if e >= 1:
                self.flashes.remove(fl)
                continue
            aacircle(arena, fl[0], 40 + 260 * e, (255, 200, 90, int(170 * (1 - e))))
        for pu in list(self.popups):
            e = (now - pu[3]) / 0.9
            if e >= 1:
                self.popups.remove(pu)
                continue
            txt = self.f_big.render(pu[2], True, pu[4])
            shd = self.f_big.render(pu[2], True, (120, 16, 22))
            a = int(255 * (1 - e ** 3))
            txt.set_alpha(a)
            shd.set_alpha(a)
            x, y = pu[0] - txt.get_width() / 2, pu[1] - 50 * e
            arena.blit(shd, (x + 3, y + 3))
            arena.blit(txt, (x, y))
        self._draw_toolbar(arena)
        self._draw_hud(arena, now)
        if self.report is None and TOOLS[self.tool][0] != "hand" and mouse[0] < PLAY_W and mouse[1] < FLOOR:
            gfxdraw.aacircle(arena, mouse[0], mouse[1], 10, (255, 255, 255))
            pygame.draw.line(arena, (255, 255, 255), (mouse[0] - 14, mouse[1]), (mouse[0] + 14, mouse[1]))
            pygame.draw.line(arena, (255, 255, 255), (mouse[0], mouse[1] - 14), (mouse[0], mouse[1] + 14))
        if self.report is not None:
            self._draw_autopsy(arena, now)
        elif self.big_view:
            self._draw_big_view(arena)

        shake = (0, 0)
        if now < self.shake_until:
            shake = (random.randint(-7, 7), random.randint(-5, 5))
        scr.blit(arena, shake)
        self._draw_brain(now)

    def _fly_state(self, now: float) -> str:
        f = self.fly
        if f.dead:
            return "dead"
        if f.grabbed is not None:
            return "grabbed"
        if now < f.escape_until:
            return "escaping!"
        if now < f.stun_until:
            return "stunned"
        return f.action

    def _text(self, surf, s, pos, color=TEXT, font=None, anchor="topleft"):
        img = (font or self.f_text).render(s, True, color)
        rect = img.get_rect(**{anchor: pos})
        surf.blit(img, rect)
        return rect

    def _draw_hud(self, surf, now: float) -> None:
        fly = self.fly
        card = pygame.Surface((236, 62), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 170), card.get_rect(), border_radius=10)
        surf.blit(card, (10, 8))
        self._text(surf, f"${self.bucks:,}", (22, 12), AMBER, self.f_head)
        self._text(surf, f"hits {self.hits}   kills {self.kills}   {self._fly_state(now)}", (22, 42), TEXT, self.f_text)
        # health
        bw, bx, by = 300, PLAY_W // 2 - 150, 16
        frac = fly.health / MAX_HEALTH
        col = S_GOOD if frac > 0.5 else S_WARN if frac > 0.25 else S_CRIT
        pygame.draw.rect(surf, (10, 12, 18), (bx - 4, by - 4, bw + 8, 26), border_radius=13)
        if frac > 0:
            pygame.draw.rect(surf, col, (bx, by, max(18, int(bw * frac)), 18), border_radius=9)
            pygame.draw.rect(surf, tuple(min(255, c + 60) for c in col), (bx + 6, by + 3, max(6, int(bw * frac) - 12), 4), border_radius=2)
        self._text(surf, "DEAD" if fly.dead else f"HEALTH {fly.health:.0f}", (PLAY_W // 2, by + 9), INK, self.f_bold, "center")
        self._text(surf, "1-6 tools   B brain   R new fly   Esc quit", (PLAY_W - 14, 14), LABEL, self.f_small, "topright")
        self._draw_pain(surf)

    def _draw_pain(self, surf) -> None:
        x, y, w, h = 10, 78, 236, 176
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 180), card.get_rect(), border_radius=10)
        surf.blit(card, (x, y))
        p = self.pain
        word, col = (("MAXED OUT", S_CRIT) if p >= 99 else ("agony", S_CRIT) if p >= 70 else ("severe", (236, 131, 90))
                     if p >= 35 else ("mild", S_WARN) if p >= 10 else ("calm", S_GOOD))
        self._text(surf, "PAIN", (x + 12, y + 8), LABEL, self.f_small)
        r = self._text(surf, f"{p:.0f}", (x + 12, y + 20), INK, self.f_title)
        self._text(surf, "/100", (r.right + 4, r.bottom - 22), LABEL, self.f_small)
        self._text(surf, word, (x + w - 12, y + 30), col, self.f_bold, "topright")
        bx, bw = x + 12, w - 24
        pygame.draw.rect(surf, (30, 36, 48), (bx, y + 64, bw, 10), border_radius=5)
        if p > 0.5:
            pygame.draw.rect(surf, col, (bx, y + 64, max(8, int(bw * p / 100)), 10), border_radius=5)
        yy = y + 84
        for label, v in zip(("touch overload", "heat sensors", "DN alarm"), self.pain_parts):
            self._text(surf, label, (bx, yy - 3), TEXT, self.f_small)
            self._bar(surf, bx + 112, yy, bw - 146, float(v), (170, 176, 188))
            self._text(surf, f"{v * 100:3.0f}", (bx + bw, yy - 3), TEXT, self.f_small, "topright")
            yy += 15
        tr = self.pain_trace
        sy, sh = y + 132, 22
        pygame.draw.rect(surf, (22, 26, 34), (bx, sy, bw, sh), border_radius=3)
        if len(tr) > 2:
            pts = [(bx + k * bw / 719, sy + sh - tr[k] / 100 * (sh - 2)) for k in range(0, len(tr), 3)]
            if len(pts) > 1:
                pygame.draw.aalines(surf, S_CRIT, False, pts)
        self._text(surf, f"peak {self.pain_peak:.0f}   maxed {self.pain_max_s:.1f}s", (bx, y + h - 17), LABEL, self.f_small)

    def _draw_toolbar(self, surf) -> None:
        bw, gap = 137, 7
        x0 = (PLAY_W - bw * len(TOOLS) - gap * (len(TOOLS) - 1)) // 2
        self.tool_rects = []
        for k, (name, label, hint) in enumerate(TOOLS):
            r = pygame.Rect(x0 + k * (bw + gap), FLOOR + 34, bw, 62)
            self.tool_rects.append(r)
            on = k == self.tool
            card = pygame.Surface(r.size, pygame.SRCALPHA)
            pygame.draw.rect(card, (60, 46, 22, 230) if on else (16, 18, 24, 210), card.get_rect(), border_radius=12)
            surf.blit(card, r)
            pygame.draw.rect(surf, AMBER if on else BORDER, r, 2, border_radius=12)
            draw_icon(surf, name, (r.x + 22, r.centery + 2), AMBER if on else TEXT)
            self._text(surf, f"{k + 1} {label}", (r.x + 42, r.y + 10), AMBER if on else INK, self.f_bold)
            self._text(surf, hint, (r.x + 42, r.y + 34), LABEL, self.f_small)

    def _draw_brain(self, now: float) -> None:
        scr, br = self.screen, self.brain
        x0 = PLAY_W
        pygame.draw.rect(scr, PANEL_BG, (x0, 0, W - x0, H))
        pygame.draw.line(scr, BORDER, (x0, 0), (x0, H))
        x = x0 + 12
        self._text(scr, "THE FLY'S BRAIN", (x, 8), ACCENT, self.f_head)
        status, scol = ("FLATLINE", S_CRIT) if br.dead else ("LIVE", S_GOOD)
        r = self._text(scr, status, (W - 14, 13), scol, self.f_bold, "topright")
        aacircle(scr, (r.x - 10, r.centery), 4, scol if br.dead or int(now * 2) % 2 else DIM)
        self._text(scr, f"MaleCNS v1.0 connectome, {br.n:,} neurons", (x, 34), LABEL, self.f_small)

        nw, nh = VIEW_SIZES["panel"]
        self.view_rect = pygame.Rect(x, 54, nw, nh)
        if self.big_view:                            # the big view is showing it; don't pay for both renders
            pygame.draw.rect(scr, (4, 5, 8), self.view_rect)
            self._text(scr, "shown in big view", self.view_rect.center, DIM, self.f_small, "center")
        else:
            scr.blit(self._view_surface("panel"), (x, 54))
            self._text(scr, "B: big view", (x + nw - 6, 54 + nh - 16), LABEL, self.f_small, "topright")
        y = 54 + nh + 12

        bw = W - x - 12
        y = self._card(scr, x, y, bw, "TOUCH NEURONS", "spikes/s per neuron", 4)
        for region, label in (("head", "head BM/JO"), ("body", "body SNta"), ("legs", "legs SNpp"), ("wing", "wing WG")):
            hz = br.hz(region)
            self._text(scr, label, (x + 8, y - 3), TEXT, self.f_small)
            self._bar(scr, x + 104, y, bw - 150, hz / 60.0, (90, 200, 120))
            self._text(scr, f"{hz:4.0f}", (x + bw - 8, y - 3), TEXT, self.f_small, "topright")
            y += 16
        y += 10
        y = self._card(scr, x, y, bw, "DESCENDING NEURONS", "x calm baseline", 7)
        for name, _, _, label in MOTOR:
            lvl = br.level(name)
            th = THRESH.get(name)
            over = th is not None and lvl > th
            self._text(scr, label, (x + 8, y - 3), AMBER if over else TEXT, self.f_small)
            self._bar(scr, x + 200, y, bw - 246, lvl / 5.0, AMBER if over else ACCENT, None if th is None else th / 5.0)
            self._text(scr, f"{lvl:4.1f}", (x + bw - 8, y - 3), TEXT, self.f_small, "topright")
            y += 16
        y += 10
        y = self._card(scr, x, y, bw, "WHOLE BRAIN", "spikes/s, last 12 s", 3)
        tr = br.history(br.hist_n - 600, br.hist_n)[:, br.col["whole brain"]]
        th_h = 40
        if len(tr) > 2:
            top = max(float(tr.max()), 8.0)
            pts = [(x + 8 + k * (bw - 16) / (len(tr) - 1), y + th_h - tr[k] / top * (th_h - 4)) for k in range(0, len(tr), 2)]
            pygame.draw.aalines(scr, ACCENT, False, pts)
            self._text(scr, f"{tr[-1]:.1f}", (x + bw - 8, y - 6), TEXT, self.f_small, "topright")
        y += th_h + 12

        self._text(scr, "REACTIONS", (x, y), LABEL, self.f_small)
        y += 16
        for t, msg in reversed(self.log[-4:]):
            age = now - t
            col = AMBER if age < 1.0 else TEXT if age < 5 else DIM
            self._text(scr, f"{age:4.1f}s  {msg}", (x, y), col, self.f_small)
            y += 15
        self._text(scr, f"sim {br.steps_per_s:4.0f} steps/s  {br.sim.last_step_ms:4.1f} ms/step",
                   (x, H - 18), DIM, self.f_small)

    def _card(self, scr, x, y, w, title, unit, rows) -> int:
        h = 24 + rows * 16
        pygame.draw.rect(scr, CARD, (x - 4, y - 4, w + 8, h + 4), border_radius=8)
        self._text(scr, title, (x + 4, y), LABEL, self.f_small)
        self._text(scr, unit, (x + w - 6, y), DIM, self.f_small, "topright")
        return y + 22

    def _bar(self, scr, x, y, w, frac, color, mark=None) -> None:
        pygame.draw.rect(scr, (30, 36, 48), (x, y, w, 8), border_radius=4)
        fw = int(w * max(0.0, min(frac, 1.0)))
        if fw > 0:
            pygame.draw.rect(scr, color, (x, y, max(fw, 4), 8), border_radius=4)
        if mark is not None:
            tx = x + int(w * min(mark, 1.0))
            pygame.draw.line(scr, INK, (tx, y - 2), (tx, y + 9))

    def _draw_autopsy(self, surf, now: float) -> None:
        rep = self.report
        e = min(1.0, (now - self.fly.dead_at - AUTOPSY_DELAY) / 0.3)
        veil = pygame.Surface((PLAY_W, H), pygame.SRCALPHA)
        veil.fill((6, 7, 10, int(200 * e)))
        surf.blit(veil, (0, 0))
        card = pygame.Rect(28, 26, PLAY_W - 56, H - 52)
        pygame.draw.rect(surf, (20, 22, 28), card, border_radius=16)
        pygame.draw.rect(surf, BORDER, card, 1, border_radius=16)
        x, y = card.x + 26, card.y + 18
        self._text(surf, "BRAIN AUTOPSY", (x, y), INK, self.f_title)
        m, s = divmod(int(rep["alive_s"]), 60)
        self._text(surf, f"survived {m}:{s:02d}   {rep['hits']} hits   killed by {rep['by']}", (x + 2, y + 44), LABEL, self.f_text)
        top = rep["top"]
        self._text(surf, f"Biggest change before death: {top['name']} at {top['ratio']:.1f}x its calm rate "
                         f"({top['base']:.1f} > {top['final']:.1f} spikes/s)", (x + 2, y + 68), TEXT, self.f_text)
        pr = self._text(surf, "PAIN", (x + 2, y + 92), S_CRIT, self.f_bold)
        self._text(surf, f"peak {rep['pain_peak']:.0f}/100    maxed out for {rep['pain_max_s']:.1f} s    "
                         f"last 2 s alive {rep['pain_last']:.0f}/100", (pr.right + 10, y + 92), TEXT, self.f_text)
        r_btn = pygame.Rect(card.right - 196, card.y + 22, 170, 42)
        pygame.draw.rect(surf, AMBER, r_btn, border_radius=10)
        self._text(surf, "NEW FLY  (R)", r_btn.center, (30, 20, 8), self.f_bold, "center")
        self.new_fly_rect = r_btn

        # diverging bars: last 2 s alive vs calm baseline, log2 scale
        y0 = card.y + 142
        lab_w, col_w = 170, 64
        cx0 = x + lab_w
        chart_w = card.w - 52 - lab_w - col_w * 3 - 40      # 40 px keeps a full-length bar's label off the columns
        mid = cx0 + chart_w // 2
        self._text(surf, "Last 2 s alive vs its calm baseline", (x, y0 - 8), INK, self.f_bold)
        hdr_y = y0 + 18
        self._text(surf, "calm", (cx0 + chart_w + col_w, hdr_y), LABEL, self.f_small, "topright")
        self._text(surf, "final", (cx0 + chart_w + col_w * 2, hdr_y), LABEL, self.f_small, "topright")
        self._text(surf, "dead", (cx0 + chart_w + col_w * 3, hdr_y), LABEL, self.f_small, "topright")
        for tick, lab in ((-4, "1/16x"), (-2, "1/4x"), (0, "same"), (2, "4x"), (4, "16x")):
            tx = mid + tick / 4 * (chart_w / 2 - 8)
            self._text(surf, lab, (tx, hdr_y), LABEL, self.f_small, "midtop")
        ry = hdr_y + 20
        row_h = 15
        chart_top = ry
        n_rows = len(rep["rows"])
        for k in (-4, -2, 2, 4):
            tx = int(mid + k / 4 * (chart_w / 2 - 8))
            pygame.draw.line(surf, (32, 35, 42), (tx, chart_top - 2), (tx, chart_top + n_rows * row_h + 8))
        pygame.draw.line(surf, (90, 90, 86), (mid, chart_top - 2), (mid, chart_top + n_rows * row_h + 8), 2)
        prev_section = None
        for row in rep["rows"]:
            if prev_section and row["section"] != prev_section:
                ry += 8
            prev_section = row["section"]
            lr = math.log2(max(row["ratio"], 1e-3))
            frac = max(-1.0, min(1.0, lr / 4))
            bl = int(abs(frac) * (chart_w / 2 - 8))
            col = UP if frac > 0 else DOWN
            if bl >= 2:
                rx = mid + 2 if frac > 0 else mid - 2 - bl
                pygame.draw.rect(surf, col, (rx, ry + 2, bl, row_h - 5), border_radius=3)
            else:
                aacircle(surf, (mid, ry + row_h // 2), 3, MID)
            self._text(surf, row["name"], (x, ry), TEXT, self.f_small)
            pct = (row["ratio"] - 1) * 100
            lab = f"{row['ratio']:.1f}x" if row["ratio"] >= 2 else f"{pct:+.0f}%"
            lx = mid + 2 + bl + 6 if frac > 0 else mid - 2 - bl - 6
            self._text(surf, lab, (lx, ry), INK, self.f_small, "topleft" if frac > 0 else "topright")
            self._text(surf, f"{row['base']:.1f}", (cx0 + chart_w + col_w, ry), LABEL, self.f_small, "topright")
            self._text(surf, f"{row['final']:.1f}", (cx0 + chart_w + col_w * 2, ry), INK, self.f_small, "topright")
            self._text(surf, f"{row['after']:.1f}", (cx0 + chart_w + col_w * 3, ry), DIM, self.f_small, "topright")
            ry += row_h
        self._text(surf, "spikes/s per neuron. Red = higher than calm, blue = lower.", (x, ry + 6), LABEL, self.f_small)

        # timeline: x calm baseline, one shared axis
        ty = ry + 30
        th = card.bottom - ty - 48
        tx0, tw = x + 40, card.w - 52 - 40
        self._text(surf, "Activity vs calm, 20 s before death to 3 s after", (x, ty - 4), INK, self.f_bold)
        lx = x + 380
        for (label, _), col in zip(rep["series"], SERIES):
            pygame.draw.rect(surf, col, (lx, ty + 3, 14, 4), border_radius=2)
            lx = self._text(surf, label, (lx + 20, ty - 2), TEXT, self.f_small).right + 18
        ty += 22
        th -= 22
        lo, hi = -3.0, 5.0                               # log2 axis: 1/8x (and silence) up to 32x

        def yof(v):
            return ty + th - (min(max(math.log2(max(float(v), 1e-6)), lo), hi) - lo) / (hi - lo) * th

        for v, lab in ((0.125, "0"), (0.25, "1/4x"), (1, "1x"), (4, "4x"), (16, "16x")):
            gy = yof(v)
            pygame.draw.line(surf, (70, 70, 66) if v == 1 else (32, 35, 42), (tx0, gy), (tx0 + tw, gy))
            self._text(surf, lab, (tx0 - 6, gy - 7), LABEL, self.f_small, "topright")
        n = max(rep["n"], 2)
        for (label, vals), col in zip(rep["series"], SERIES):
            if len(vals) < 2:
                continue
            pts = [(tx0 + k * tw / (n - 1), yof(v)) for k, v in enumerate(vals)]
            pygame.draw.lines(surf, col, False, pts, 2)
        # pain strip on the same time axis: one hue, dark = none, bright red = maxed
        sy = ty + th + 8
        self._text(surf, "pain", (tx0 - 6, sy), LABEL, self.f_small, "topright")
        pain = rep["pain"]
        if len(pain):
            step = max(1, len(pain) // 300)
            cw = tw / len(pain) * step
            for k in range(0, len(pain), step):
                t = float(np.clip(pain[k] / 100, 0, 1))
                col = tuple(int(MID[j] + (UP[j] - MID[j]) * t) for j in range(3))
                pygame.draw.rect(surf, col, (tx0 + k * tw / len(pain), sy, math.ceil(cw) + 1, 14))
        dx = tx0 + rep["death_at"] * tw / (n - 1)
        pygame.draw.line(surf, S_CRIT, (dx, ty - 2), (dx, sy + 14), 2)
        self._text(surf, "died", (dx + 6, ty), S_CRIT, self.f_small)

    def handle(self, ev, now: float) -> bool:
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            return False
        if ev.type == pygame.KEYDOWN:
            if pygame.K_1 <= ev.key < pygame.K_1 + len(TOOLS):
                self.tool = ev.key - pygame.K_1
            elif ev.key == pygame.K_r:
                self.new_fly()
            elif ev.key == pygame.K_b:
                self.big_view = not self.big_view
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.report is not None:
                if getattr(self, "new_fly_rect", None) and self.new_fly_rect.collidepoint(ev.pos):
                    self.new_fly()
                return True
            if self.view_rect.collidepoint(ev.pos):
                self.big_view = not self.big_view
                return True
            for k, r in enumerate(getattr(self, "tool_rects", [])):
                if r.collidepoint(ev.pos):
                    self.tool = k
                    return True
            if ev.pos[0] < PLAY_W:
                self.use_tool(ev.pos, now)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.fly.grabbed = None
            self.torching = False
        return True


def load_brain(out: dict) -> None:
    try:
        import brainpack
        from connectome.sim import LIFParams, LIFSim

        pack = brainpack.find()
        if pack is None:                             # source checkout: pack the connectome once (~30 s)
            out["stage"] = "building the brain pack (first run only)"
            pack = brainpack.build()
        out["stage"] = "unpacking the fly's brain"
        g, weights, soma = brainpack.load(pack)
        out["stage"] = f"wiring {g.n:,} neurons"
        sim = LIFSim(None, LIFParams(), W_in=weights)
        out["stage"] = "placing neurons"
        out["view"] = BrainView(soma, weights)
        brain = Brain(g, sim)
        out["stage"] = "waking the fly up"
        brain.warmup()
        out["brain"] = brain
    except Exception as e:  # shown on the loading screen
        out["error"] = f"{type(e).__name__}: {e}"


def main() -> int:
    smoke = float(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0.0  # build check: run N s, exit
    pygame.init()
    pygame.display.set_caption("Kick the Fly")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui,consolas", 22)
    state: dict = {"stage": "starting"}
    threading.Thread(target=load_brain, args=(state,), daemon=True).start()
    t0 = time.perf_counter()
    while "brain" not in state:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return 0
        screen.fill(BG)
        msg = state.get("error") or f"{state['stage']}{'.' * (int((time.perf_counter() - t0) * 3) % 4)}"
        img = font.render(msg, True, RED if "error" in state else TEXT)
        screen.blit(img, img.get_rect(center=(W // 2, H // 2)))
        pygame.display.flip()
        clock.tick(30)

    brain = state["brain"]
    brain.start()
    game = Game(screen, brain, state["view"])
    running = True
    t_game = time.perf_counter()
    while running:
        now = time.perf_counter()
        if smoke and now - t_game > smoke:
            status = f"smoke ok: {brain.n:,} neurons, {brain.steps_per_s:.0f} steps/s"
            print(status)
            if len(sys.argv) > sys.argv.index("--smoke") + 2:   # optional screenshot path; the exe has no console
                shot = sys.argv[sys.argv.index("--smoke") + 2]
                pygame.image.save(screen, shot)
                with open(shot + ".txt", "w") as f:
                    f.write(status)
            break
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            running = game.handle(ev, now) and running
        game.update(now, (min(mouse[0], PLAY_W - 5), mouse[1]))
        game.draw(now, mouse)
        pygame.display.flip()
        clock.tick(60)
    brain.stop()
    pygame.quit()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        if getattr(sys, "frozen", False):            # no console in the exe: leave the traceback next to it
            import traceback
            from pathlib import Path

            (Path(sys.executable).resolve().parent / "KickTheFly-crash.txt").write_text(traceback.format_exc())
        raise
