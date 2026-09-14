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
partners, colored by direction and shaded by depth. Neurons firing above their
calm rate glow: pain-sensing neurons hot orange, the rest cyan, and strongly
firing ones sparkle. Real neuron shapes aren't bundled, so the fibers are
estimates. B toggles a big view.

Pain neurons (P): the connectome can't be given extra neurons, so this setting
listens to more of the fly's real ones and makes each hit fire more of them.
  normal  9,080 neurons   touch, heat/cold, smell and taste groups; a light touch
                          fires 30% of a region's neurons
  more   11,392 neurons   + the rest of the body's sensory neurons (campaniform
                          sensilla, hair plates, chordotonal organs...), which
                          also fire at half strength on every touch; 60%
  max    13,238 neurons   + the 1,846 ascending neurons that carry body signals
                          to the brain (the "body relay" pain part); 100%

Immortal (I): it still feels everything, but health stops at 1 and it heals when
left alone, melting and freezing stop short and wear off, and it breaks out of
spider silk.

It sees you coming: the game measures how fast each object (your tool, the
swatter, the spider, bombs) grows in the fly's view and drives its real looming
detectors LPLC2 and LC4 with that. Streaming pixels through the sim's own
photoreceptors didn't work: a looming image stayed inside the brain's random
flicker. Driven directly, LPLC2/LC4 excite the giant fiber DNp01 to 7-12x calm
(its calm max is 2.6x), and DNp01 above 4x makes the fly DODGE. Move slowly and
it won't see you.

Brain surgery (O): silence or stimulate real neuron groups with a constant
current. The inspector (click a neuron in the big view) shows its type, firing,
and strongest partners from the connectome, and can silence or stimulate its type.

Learning (a rule added on top of the fixed connectome): each tool carries a
scent, 6 of the 53 olfactory glomeruli, that the fly smells within 330 px. The
smell reaches its Kenyon cells for real (repeat scents give correlated Kenyon
cell patterns, 0.82 vs 0.38 between scents). Pain drives the PPL1 punishment
dopamine neurons (game rule), and while they fire, active Kenyon cells gain
"fear"; PAM reward neurons give "liking" the same way. When a remembered
tool's scent comes close, the fly runs or flies away. Silence PPL1 or the
Kenyon cells in surgery and it can't learn.

Arenas (E): the fan's wind drives Johnston's organ wind neurons (JO-C/E); the
pool's water drives humidity neurons (HRN) and it floats, gets wet wings and
can drown; flypaper glues any part that touches it, and the stuck fly's leg
neurons fire; the lamp's light drives photoreceptors and it is drawn to it
(phototaxis is a game rule), and the hot bulb fires heat sensors.

Tracking: every neuron's spikes are counted into its group each 5 ms step. Each
group's rate is compared with its own calm baseline, a 10 s average taken only
while nothing has touched the fly for 2 s, so hits don't inflate "normal".

Pain: the adult connectome has no neurons annotated as nociceptors, so the PAIN
index is a game estimate built from real signals, not a measurement of what the
fly feels. It adds these, capped at 100:
  55%  touch overload  all touch neurons above calm, saturating at +35 spikes/s
  25%  heat / cold     hot cells TRN_VP2 or cold cells TRN_VP3 above calm, +30
  55%  chemical        smell ORNs and taste neurons above calm, +35
  20%  DN alarm        all descending neurons more than 6% above calm, saturating
                       at +36%, held at its peak for ~1 s
  30%  body relay      (max pain setting only) ascending neurons above calm, the
                       same way
The blowtorch drives every touch and heat neuron at full strength, which pins it.

More ways to kill it (game rules around real sensory input):
  zapper   every touch neuron plus current into a random 30% of all neurons for 40 ms
  freeze   cold cells TRN_VP3 and a little touch; cooling damps the brain; frozen solid,
           then any hit shatters it
  spider   hunts it; bites hit body and leg touch neurons, venom paralyses and damps the
           brain, two bites wrap it in silk

Reward: sugar makes the fly walk over and eat (game rule). Eating drives taste
neurons and the PAM dopaminergic neurons (316), which signal reward in flies. In
this sim taste input alone doesn't reach PAM, so sugar drives them directly, as in
PAM activation experiments. The REWARD meter is PAM activity above calm.

Flight: DNg02 (29 wing-power descending neurons). It takes off when DNg02 fires
above 1.58x calm, and flies away when the head-touch escape group fires. Flight
speed follows DNg02. The flight path itself is a game rule.

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
import os
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

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
SENSE = {"wind": ("JO-C", "JO-E"), "head": ("BM_", "JO-"), "body": ("SNta",), "legs": ("SNpp",), "wing": ("WG",),
         "heat": ("TRN_VP2",), "cold": ("TRN_VP3",), "humid": ("HRN_",), "smell": ("ORN_",),
         "taste": ("LgLG", "LgAG", "LB", "PhG", "claw_"), "light": ("R1-R6", "R7", "R8"),
         "loom": ("LPLC2", "LC4")}               # wind goes first so it keeps JO-C/E; head keeps the rest of JO
NOT_SENSORY = {"loom"}                          # LPLC2/LC4 are visual projection neurons, not sensory neurons
TOUCH = ("head", "body", "legs", "wing")
MOTOR = (  # name, types, side, label
    ("jump", ("DNg85", "DNg48", "DNg37", "DNge067", "DNg29", "DNge132"), None, "head-touch DNs  > JUMP"),
    ("run", ("DNge122", "DNge104", "DNge102", "DNxl114", "DNge182", "DNge048"), None, "body-touch DNs  > RUN"),
    ("kick", ("DNge074", "DNge075", "DNge096", "DNg30", "DNg34", "DNp38"), None, "leg-touch DNs   > KICK"),
    ("walk", ("DNp09",), None, "DNp09 (P9)      > WALK"),
    ("back", ("MDN",), None, "MDN moonwalker  > BACK UP"),
    ("turn_l", ("DNa01", "DNa02"), "L", "DNa01/02 left   > TURN"),
    ("turn_r", ("DNa01", "DNa02"), "R", "DNa01/02 right  > TURN"),
    ("fly", tuple(f"DNg02_{c}" for c in "abcdefg"), None, "DNg02 wing power> FLY"),
    ("escape", ("DNp01",), None, "DNp01 giant fiber> DODGE"),
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
# DNg02 (29 wing-power DNs) rests at ~7 spikes/s; over 30 s of play its level never passed 1.64x (p99.9 1.60), so 1.58x
# takes off now and then.
# DNp01, the giant fiber: over 60 s calm its level never passed 2.64x; driving the looming detectors took it to 7-12x.
THRESH = {"jump": 3.0, "run": 2.4, "kick": 2.0, "walk": 3.0, "back": 3.8, "turn": 2.1, "fly": 1.58, "escape": 4.0}
PAIN_WEIGHTS = np.array([0.55, 0.25, 0.55, 0.20, 0.30])   # touch, thermal, chemical, DN alarm, body relay; cap 100
PAIN_LEVELS = (  # name, share of a region's neurons a light touch recruits, how hard the rest of the body's sensors join
    ("normal", 0.3, 0.0), ("more", 0.6, 0.5), ("max", 1.0, 1.0),
)
SURGERY_CURRENT = {-1: -0.6, 0: 0.0, 1: 0.12}   # x ext_gain 4: silenced -2.4 per step (beats any touch), stimulated +0.48
TOOL_NAMES = ("hand", "flick", "swatter", "bomb", "torch", "cleaner", "zapper", "freeze", "spider", "sugar")
STIM_AMP = 0.5              # x ext_gain 4 = 2.0 per step: a driven neuron fires every refractory cycle
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
            m = add_detail(region, m if region in NOT_SENSORY else m & (np.char.find(sc, "sensory") >= 0))
            self.sense[(region, None)] = np.flatnonzero(m)
            for s in "LR":
                rows = np.flatnonzero(m & (sides == f"_{s}"))
                self.sense[(region, s)] = rows if len(rows) > 0.2 * m.sum() else np.flatnonzero(m)
        for name, tys, side, _ in MOTOR:
            m = is_dn & np.isin(types, tys)
            if side:
                m &= sides == f"_{side}"
            add_detail(name, m)
        # the rest of the body's sensory neurons (campaniform sensilla, hair plates, chordotonal organs, unnamed SN*):
        # only counted and driven when the pain setting asks for more neurons
        body_extra = add_detail("body_extra", np.isin(sc, ("vnc_sensory", "vnc_sensory_tbc", "sensory_ascending",
                                                            "sensory_ascending_tbc")))
        self.sense[("body_extra", None)] = np.flatnonzero(body_extra)
        self.recruit, self.spill, self.pain_level = 0.3, 0.0, 0
        # PAM dopaminergic neurons (316) signal reward in flies; sugar drives them directly (see the docstring)
        self.sense[("reward", None)] = np.flatnonzero(add_detail("reward", np.char.startswith(types, "PAM")))
        # PPL1 dopaminergic neurons (16) signal punishment; pain drives them (game rule), learning reads them
        self.sense[("punish", None)] = np.flatnonzero(add_detail("punish", np.char.startswith(types, "PPL1")))
        self.kc = np.flatnonzero(np.char.startswith(types, "KC"))        # Kenyon cells, the mushroom body
        orn = self.sense[("smell", None)]
        glom = np.array([t.split("_", 1)[1] for t in types[orn]])
        order = np.random.default_rng(11).permutation(np.unique(glom))
        for k, name in enumerate(TOOL_NAMES):        # each tool's scent: its own 5 of the 53 olfactory glomeruli
            self.sense[("scent", name)] = orn[np.isin(glom, order[k * 5:(k + 1) * 5])]
        self.types, self.superclass = types, sc
        self.instance = np.array([i or "" for i in g.instance])
        self.override = np.zeros(g.n, np.float32)    # brain surgery: per-neuron silencing / stimulating current
        self.surgery = False
        self.sense[("all", None)] = np.arange(g.n)   # the zapper's shock
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

    def poke(self, region: str, side: str | None, strength: float, recruit: float | None = None) -> None:
        """Drive a random share of a region's neurons; harder hits recruit more of them for longer."""
        if self.dead:
            return
        s = float(np.clip(strength, 0, 1))
        pop = self.sense[(region, side)]
        share = self.recruit + (1 - self.recruit) * s if recruit is None else recruit
        rows = self.rng.choice(pop, size=max(1, int(len(pop) * share)), replace=False)
        steps = 8 + int(40 * s)
        with self._lock:
            self.last_poke = self.steps
            old = self._pending.get((region, side))
            if old is None or steps > old[1]:
                self._pending[(region, side)] = [rows, steps]
        if region in TOUCH and self.spill > 0:       # more pain neurons: the rest of the body's sensors fire too
            self.poke("body_extra", None, s * self.spill)

    def set_override(self, rows: np.ndarray, mode: int) -> None:
        """Brain surgery on these neurons: -1 silence, 0 leave alone, +1 stimulate."""
        self.override[rows] = SURGERY_CURRENT[mode]
        self.surgery = bool(np.any(self.override))

    def clear_overrides(self) -> None:
        self.override[:] = 0
        self.surgery = False

    def set_pain_level(self, level: int) -> None:
        _, self.recruit, self.spill = PAIN_LEVELS[level]
        self.pain_level = level

    def pain_neurons(self, level: int | None = None) -> int:
        """How many real neurons the pain index listens to at this setting."""
        level = self.pain_level if level is None else level
        names = list(TOUCH) + ["heat", "cold", "smell", "taste"] + (["body_extra"] if level >= 1 else [])
        names += ["ascending"] if level >= 2 else []
        return int(sum(self.g_size[self.col[n]] for n in names))

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
        if self.death_step is not None:                  # start from any damping already on, or it would rebound
            ramp = min(1.0, max(self.sedation, (self.steps - self.death_step) / 300))
            spikes = self.sim.step(self._inhib * np.float32(ramp))
        else:
            drive = self._cur
            if self.sedation > 0:                    # solvent, cold or venom depression
                drive = drive + self._inhib * np.float32(self.sedation)
            if self.surgery:
                drive = drive + self.override
            spikes = self.sim.step(drive if (active or self.sedation > 0 or self.surgery) else None)
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
        if self.death_step is None and self.sedation == 0 and not self.surgery and self.steps - self.last_poke > CALM_STEPS:
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
        """(touch overload, thermal, chemical, DN alarm, body relay) in 0..1 per row of group rates. See the docstring."""
        rates = np.atleast_2d(rates)

        def above(names, full):
            idx = [self.col[n] for n in names]
            sz = self.g_size[idx]
            return ((rates[:, idx] - base[idx]) * sz).sum(1) / sz.sum() / full

        touch = above(list(TOUCH) + (["body_extra"] if self.pain_level else []), 35.0)
        thermal = np.maximum(above(["heat"], 30.0), above(["cold"], 30.0))   # thermosensors fire ~24 spikes/s at rest
        chemical = (above(["smell", "taste"], 1.0) - 8.0) / 27.0   # 8 spikes/s deadzone: a tool's faint scent isn't pain
        d = self.col["descending"]
        alarm = (rates[:, d] / max(float(base[d]), 1.0) - 1.06) / 0.3        # 6% deadzone: resting DN flicker
        a = self.col["ascending"]                    # ascending neurons carry body signals up to the brain
        relay = (rates[:, a] / max(float(base[a]), 1.0) - 1.06) / 0.35 if self.pain_level >= 2 else np.zeros(len(rates))
        return np.clip(np.stack([touch, thermal, chemical, alarm, relay], 1), 0, 1)

    @staticmethod
    def hold_alarm(parts: np.ndarray, decay: float = 0.95) -> np.ndarray:
        """Peak-hold the DN alarm (~1 s at one sample per 20 ms): the gain controller makes the raw surge flicker."""
        out = parts.copy()
        for k in range(1, len(out)):
            out[k, 3] = max(out[k, 3], out[k - 1, 3] * decay)
        return out

    @staticmethod
    def pain_index(parts: np.ndarray) -> np.ndarray:
        return np.minimum(100.0, 100.0 * (parts @ PAIN_WEIGHTS))

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
    neurons) are just the tuft. Real neuron shapes aren't in the pack, so fibers are estimates. The wiring is colored by
    fiber direction (red left-right, green up-down, blue front-back) and shaded by depth, front brightest. Firing above
    a neuron's own calm rate glows: pain-sensing neurons (touch, heat, cold, chemical, and the ascending relay) glow
    hot orange, everything else cool white, and firing neurons that spiked on the latest step sparkle at their cell
    body. The nerve cord is cut off at the neck, so ascending fibers enter through it.
    """

    FIBER, ARBOR = 18, 3
    HOT = np.array([1.0, 0.34, 0.07], np.float32)

    def __init__(self, soma: np.ndarray, W, pain_mask: np.ndarray, seed: int = 1):
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
        self.hot_mask = np.asarray(pain_mask, bool)
        cool = (0.35 * self.col + 0.65 * np.array([0.3, 0.75, 1.0], np.float32)) * 0.55   # everything else: dim cyan
        self.tint = np.where(self.hot_mask[:, None], self.HOT * 2.2, cool).astype(np.float32)

        t = np.linspace(0, 1, self.FIBER, dtype=np.float32)[None, :, None]
        bend = rng.normal(size=(n, 3)).astype(np.float32) * (0.12 * length)[:, None]
        mid = (start + A) / 2 + bend
        fiber = (1 - t) ** 2 * start[:, None] + 2 * (1 - t) * t * mid[:, None] + t ** 2 * A[:, None]
        r = np.clip(0.05 * length, 400, 1600)
        arbor = A[:, None] + rng.normal(size=(n, self.ARBOR, 3)).astype(np.float32) * r[:, None, None]
        pts = np.concatenate([fiber, arbor], 1)                      # (n, samples, 3)
        wts = np.concatenate([np.ones(self.FIBER), np.full(self.ARBOR, 2.0)]).astype(np.float32)
        depth = np.clip(1.2 - (pts[..., 2] - 5000.0) / 50000.0, 0.35, 1.0)   # the front of the brain is brighter
        nid = np.broadcast_to(np.arange(n)[:, None], pts.shape[:2])
        keep = ok[:, None] & (pts[..., 2] < VIEW_ZCUT)

        self.M, self.base, self.gain, self.spark_pix = {}, {}, {}, {}
        for key, (w, h) in VIEW_SIZES.items():
            px = ((pts[..., 0] - VIEW_X[0]) / (VIEW_X[1] - VIEW_X[0]) * w).astype(np.int32)
            py = ((pts[..., 1] - VIEW_Y[0]) / (VIEW_Y[1] - VIEW_Y[0]) * h).astype(np.int32)
            m = keep & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            P = sp.coo_array(((wts * depth)[m], ((py * w + px)[m], nid[m])), shape=(w * h, n)).tocsr()
            self.M[key] = P
            struct = P @ self.col
            p99 = float(np.percentile(struct.max(1), 99.0)) or 1.0
            self.base[key] = struct * (0.5 / p99)
            self.gain[key] = 3.4 / p99
            self.spark_pix[key] = np.where(m[:, 0], py[:, 0] * w + px[:, 0], -1)   # cell body (or arbor) pixel
        self.calm = np.full(n, 0.025, np.float32)                   # per-neuron calm rate, spikes per step
        self.firing = self.hot_firing = 0

    def render(self, key: str, rates: np.ndarray, spiked: np.ndarray, t: float, learn: bool) -> pygame.Surface:
        if learn:
            self.calm += (rates - self.calm) * 0.01
        excess = np.maximum(rates / 0.025 - self.calm / 0.025 - np.where(self.hot_mask, 1.0, 0.6), 0)
        act = (2.2 * excess).astype(np.float32)
        firing = act > 0.05
        self.firing, self.hot_firing = int(firing.sum()), int((firing & self.hot_mask).sum())
        light = (self.M[key] @ (act[:, None] * self.tint)) * self.gain[key]
        if len(spiked):                               # sparkles: firing neurons that spiked on the latest step
            s = spiked[act[spiked] > 2.0]
            pix = self.spark_pix[key][s]
            s, pix = s[pix >= 0], pix[pix >= 0]
            np.add.at(light, pix, np.where(self.hot_mask[s, None], self.HOT * 3.0, np.float32(0.7)))
        w, h = VIEW_SIZES[key]
        img = (255 * (1 - np.exp(-(self.base[key] + light)))).astype(np.uint8)
        surf = pygame.image.frombuffer(img.tobytes(), (w, h), "RGB").copy()
        hot = (255 * (1 - np.exp(-0.7 * light))).astype(np.uint8)          # bloom from the firing only
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
ARENAS = ("room", "fan", "flypaper", "pool", "lamp")
WATER_Y = FLOOR - 140                 # pool surface
PAPER_X = (230.0, 660.0)              # flypaper strip on the floor
LAMP = (445.0, 196.0)                 # bulb center
FAN = (70.0, 548.0)                   # fan hub, low on the left so its wind hits a standing fly


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
        self.hover = np.array([x, 300.0])   # flight: current hover point and where it's heading
        self.fly_target = np.array([x, 300.0])
        self.wander = False
        self.power = 1.0                  # DNg02 level while flying
        self.frost = 0.0                  # freeze spray, 1 = frozen solid
        self.frozen_at: float | None = None
        self.shattered_at: float | None = None
        self.venom = 0.0                  # spider venom, paralyses
        self.wrapped = False              # in the spider's silk
        self.zap_until = 0.0
        self.eating_until = 0.0
        self.arena = "room"
        self.wind = 0.0                   # fan push, px/frame^2
        self.wet = 0.0                    # seconds until its wings dry
        self.stuck: dict[int, np.ndarray] = {}   # flypaper: particle -> glued position

    @property
    def flying(self) -> bool:
        return self.escape_until > 0 and self.grabbed is None and not self.dead and not self.wrapped \
            and self.melt < 0.3 and self.frost < 0.5 and self.venom < 0.5 and self.wet <= 0 and len(self.stuck) < 2

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
        self.escape_until = min(self.escape_until, now)     # knocked out of the air

    def escape(self, now: float, seconds: float = 2.4, wander: bool = False) -> None:
        """Take off. Escapes fly up and away from the last hit; wandering flights drift around the room."""
        away = 1.0 if self.p[THX, 0] >= self.last_hit_x else -1.0
        self.prev[:] = self.p - np.array([away * 4.0, -9.0])
        self.hover = self.p[THX].copy()
        self.wander = wander
        self._new_target(away)
        self.escape_until = now + seconds
        self.escape_ready = now + seconds + 1.0

    def _new_target(self, away: float | None = None) -> None:
        x = self.p[THX, 0]
        if away is None:
            tx = random.uniform(120, PLAY_W - 120)
        else:
            tx = x + away * random.uniform(220, 380)
            if not 100 < tx < PLAY_W - 100:
                tx = x - away * random.uniform(220, 380)
        self.fly_target = np.array([float(np.clip(tx, 100, PLAY_W - 100)), random.uniform(150, 430)])

    def step(self, now: float, mouse) -> list[tuple[int, float]]:
        """One 60 Hz physics frame. Returns (particle, impact speed) for hard contacts with the arena."""
        if self.frozen_at is not None:                       # a block of ice: nothing moves until it shatters
            self.prev = self.p.copy()
            return []
        flying = now < self.escape_until and self.flying
        if flying:
            self.recover = 1.0
        elif self.grabbed is not None or now < self.stun_until or self.dead or self.wrapped:
            self.recover = 0.0
        else:
            self.recover = min(1.0, self.recover + 1 / 30)
        height = (FLOOR - STAND) - self.p[THX, 1]
        stiff = (1 - self.melt) ** 0.5 * (1 - self.frost) * (1 - self.venom)   # dissolving, freezing, paralysed
        strength = self.recover * (1.0 if flying else float(np.clip(1 - height / 160, 0, 1)) * stiff)
        if self.grabbed is not None or (self.arena == "pool" and not flying and self.p[THX, 1] > WATER_Y - 30):
            strength = 0.0                                   # held, or floating: nothing to stand on
        shrink = 1 - 0.35 * self.melt                        # dissolving: the body shrinks and slumps

        v = (self.p - self.prev) * (0.992 - 0.12 * strength)  # standing legs also damp the body
        speed = np.hypot(*v.T)
        v *= np.minimum(1.0, 60.0 / np.maximum(speed, 1e-6))[:, None]
        self.prev = self.p.copy()
        self.p += v
        self.p[:, 1] += 0.9 * (1 - strength)                 # standing or flying: legs or wings carry the weight
        if self.arena == "pool":                             # water: buoyancy and drag, so it floats at the surface
            sub = self.p[:, 1] > WATER_Y
            if sub.any():
                self.p[sub, 1] -= 1.35
                self.p[sub] -= (self.p[sub] - self.prev[sub]) * 0.12
        if self.wind:                                        # the fan pushes wings hardest
            push = np.full(N_P, self.wind * 0.35)
            push[list(WING)] *= 2.5
            self.p[:, 0] += push
            if flying:
                self.hover[0] += self.wind * 3.0

        if flying:
            self._fly(now)
        elif strength > 0:
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
            for i, at in self.stuck.items():
                self.p[i] = at
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
        for i, at in self.stuck.items():
            self.p[i] = at
            self.prev[i] = at
        if self.arena == "flypaper" and self.grabbed is None:   # anything touching the strip sticks
            for i in range(N_P):
                if i not in self.stuck and PAPER_X[0] < self.p[i, 0] < PAPER_X[1] and self.p[i, 1] >= FLOOR - RADIUS[i] - 1.5:
                    self.stuck[i] = self.p[i].copy()
        self.hurt = max(0.0, self.hurt - 1 / 20)
        return self._contacts()

    def _fly(self, now: float) -> None:
        """Hovering flight: the body is held in shape around a hover point that chases a target, legs tucked."""
        d = self.fly_target - self.hover
        dist = float(np.hypot(*d))
        speed = 2.5 + 2.5 * min(self.power, 2.0)
        if dist < 25:
            self._new_target()
        else:
            self.hover += d / dist * min(speed, dist)
        if abs(d[0]) > 8:
            self.facing = 1 if d[0] > 0 else -1
        self.anchor_x = float(self.hover[0])
        self.action = "flying"
        bob = 4 * math.sin(now * 9)
        off = REST.copy()
        off[:, 0] *= self.facing
        tgt = off + (self.hover[0], self.hover[1] + bob)
        for i in range(6):
            tgt[FOOT[i]] = tgt[KNEE[i]] + (-6 * self.facing, 14)          # legs hang tucked under the body
        d = tgt - self.p
        self.p += d * 0.25
        self.prev += d * 0.2

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


def shade(col, hurt: float, dead: bool, char: float = 0.0, melt: float = 0.0, frost: float = 0.0):
    if char:
        col = tuple(c * (1 - 0.75 * char) + 22 * char for c in col)
    if frost:
        col = tuple(c * (1 - 0.7 * frost) + t * 0.7 * frost for c, t in zip(col, (175, 215, 245)))
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
    if fly.shattered_at is not None:
        return                                        # the shards are drawn by the game
    _draw_fly_body(surf, fly, now)
    p = fly.p
    if fly.wrapped:                                   # spider silk
        axis = p[HEAD] - p[ABD]
        ang = math.atan2(axis[1], axis[0])
        mid = (p[HEAD] + p[ABD]) / 2
        L = float(np.hypot(*axis))
        gfxdraw.filled_polygon(surf, [(int(x), int(y)) for x, y in ellipse_pts(mid, L / 2 + 26, 26, ang)], (235, 235, 230, 170))
        for k in range(-3, 4):
            q = mid + np.array([math.cos(ang), math.sin(ang)]) * k * 11
            pts = ellipse_pts(q, 5, 27, ang, 16)
            gfxdraw.aapolygon(surf, [(int(x), int(y)) for x, y in pts], (250, 250, 245, 220))
    if fly.frozen_at is not None:                     # ice block
        lo, hi = p.min(0) - 14, p.max(0) + 14
        box = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
        gfxdraw.filled_polygon(surf, [(int(x), int(y)) for x, y in box], (170, 215, 245, 90))
        gfxdraw.aapolygon(surf, [(int(x), int(y)) for x, y in box], (225, 245, 255, 230))
        gfxdraw.line(surf, int(lo[0] + 8), int(lo[1] + 6), int(lo[0] + 30), int(lo[1] + 6), (255, 255, 255, 220))
    elif fly.frost > 0.1:                             # frost crystals
        rng = random.Random(7)
        for _ in range(int(12 * fly.frost)):
            i = rng.choice((HEAD, THX, ABD, ABD))
            x, y = p[i] + (rng.uniform(-24, 24), rng.uniform(-22, 18))
            for a in (0, 1.05, 2.1):
                gfxdraw.line(surf, int(x - 4 * math.cos(a)), int(y - 4 * math.sin(a)), int(x + 4 * math.cos(a)),
                             int(y + 4 * math.sin(a)), (235, 250, 255, 220))
    if now < fly.eating_until:                        # happy hearts
        for k in range(3):
            ph = (now * 0.9 + k / 3) % 1.0
            hx, hy = p[HEAD, 0] + (k - 1) * 18, p[HEAD, 1] - 30 - 40 * ph
            col = (255, 110, 150, int(255 * (1 - ph)))
            aacircle(surf, (hx - 3, hy), 4, col)
            aacircle(surf, (hx + 3, hy), 4, col)
            aapoly(surf, [(hx - 7, hy + 1), (hx + 7, hy + 1), (hx, hy + 9)], col)


def _draw_fly_body(surf: pygame.Surface, fly: Fly, now: float) -> None:
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

    zapped = now < fly.zap_until and int(now * 40) % 2 == 0

    def c(col):
        if zapped:                                    # electrocution flash
            return (200, 235, 255)
        return shade(col, hurt, dead, fly.char, fly.melt, fly.frost)

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
    elif name == "zapper":
        gfxdraw.aaellipse(surf, int(x), int(y - 4), 9, 11, col)
        thick_line(surf, (x, y + 7), (x, y + 14), 3, col)
        aapoly(surf, [(x + 1, y - 12), (x - 5, y - 2), (x, y - 2), (x - 2, y + 5), (x + 5, y - 5), (x, y - 5)], (255, 230, 90))
    elif name == "freeze":
        for a in (0, 1.047, 2.094):
            thick_line(surf, (x - 11 * math.cos(a), y - 11 * math.sin(a)), (x + 11 * math.cos(a), y + 11 * math.sin(a)), 2.5, (170, 220, 255))
    elif name == "spider":
        for sgn in (-1, 1):
            for k in range(4):
                a = (k - 1.5) * 0.45
                thick_line(surf, (x, y), (x + sgn * 13 * math.cos(a), y + 9 * math.sin(a) - 3 + k), 1.6, col)
        aacircle(surf, (x, y + 1), 6, col)
        aacircle(surf, (x, y - 6), 4, col)
    elif name == "sugar":
        aapoly(surf, [(x - 8, y - 4), (x + 2, y - 9), (x + 11, y - 4), (x + 1, y + 1)], (250, 250, 255))
        aapoly(surf, [(x - 8, y - 4), (x + 1, y + 1), (x + 1, y + 12), (x - 8, y + 7)], (215, 215, 225))
        aapoly(surf, [(x + 1, y + 1), (x + 11, y - 4), (x + 11, y + 7), (x + 1, y + 12)], (190, 190, 205))
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
TOOLS = (("hand", "HAND", "drag the fly and throw it"), ("flick", "FLICK", "click near the fly"),
         ("swatter", "SWAT", "click on the fly"), ("bomb", "BOMB", "click to drop a bomb, 1.5 s fuse"),
         ("torch", "TORCH", "hold: burns it, maxes out pain"), ("cleaner", "CLEANER", "hold: brake cleaner dissolves it"),
         ("zapper", "ZAP", "click: electric shock through its body"), ("freeze", "FREEZE", "hold: freezes it solid, then smash the ice"),
         ("spider", "SPIDER", "click: drop a spider that hunts it"), ("sugar", "SUGAR", "click: drop sugar to reward it"))
assert tuple(t[0] for t in TOOLS) == TOOL_NAMES
TOOL_KEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0)
TORCH_KEYS = (("head", None), ("body", None), ("legs", "L"), ("legs", "R"), ("wing", "L"), ("wing", "R"), ("heat", None))
OUCH = ("BONK!", "OOF!", "SPLAT!", "THWACK!", "BZZT!", "OW!")
CURSOR_SIZE = {"flick": 12, "swatter": 38, "bomb": 16, "torch": 18, "cleaner": 22, "zapper": 16, "freeze": 22, "spider": 20}
LOOM_MIN, LOOM_FULL = 1.5, 8.0        # rad/s of angular expansion: below LOOM_MIN nothing, LOOM_MIN + LOOM_FULL = full drive
SCENT_RANGE = 330.0                   # px: how close a tool must be for the fly to smell it
LEARN_RATE = 0.02                     # per frame at full dopamine, for an averagely active Kenyon cell
FEAR_ACT, LIKE_ACT = 0.35, 0.35       # learned memory that changes behavior
KC_SPARSE = 280                       # Kenyon cells kept per pattern (of 4,064)
GIF_SIZE, GIF_FRAMES = (400, 238), 90  # 15 fps x 6 s
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


class Sound:
    """Every sound is synthesized here with numpy at startup, so the game ships no audio files. M mutes."""

    RATE = 22050

    def __init__(self):
        self.ok, self.muted = False, False
        self.loops: dict[str, tuple[str, object]] = {}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(self.RATE, -16, 1, 512)
            self.channels = pygame.mixer.get_init()[2]
            pygame.mixer.set_num_channels(20)
            self.fx = self._make()
            self.ok = True
        except Exception:                            # no audio device: play silently
            self.fx = {}

    def _snd(self, x: np.ndarray, vol: float = 0.6):
        a = (np.clip(x, -1, 1) * 32767 * vol).astype(np.int16)
        if self.channels == 2:
            a = np.repeat(a[:, None], 2, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(a))

    def _make(self) -> dict:
        R = self.RATE
        rng = np.random.default_rng(5)

        def t(d):
            return np.arange(int(R * d)) / R

        def noise(d):
            return rng.uniform(-1, 1, int(R * d))

        def low(x, k):                               # moving-average lowpass
            return np.convolve(x, np.ones(k) / k, mode="same")

        def sweep(f0, f1, d):
            tt = t(d)
            f = f0 + (f1 - f0) * tt / d
            return np.sin(2 * np.pi * np.cumsum(f) / R)

        fx = {}
        for k, f in enumerate((150, 190, 240, 300)):   # wing buzz loops; 0.5 s holds whole cycles, so they loop cleanly
            tt = t(0.5)
            saw = 2 * ((tt * f) % 1) - 1
            fx[f"buzz{k}"] = self._snd(0.45 * saw * (0.8 + 0.2 * np.sin(2 * np.pi * 2 * tt)), 0.35)
        tt = t(0.18)
        fx["whack"] = self._snd(low(noise(0.18), 6) * np.exp(-tt * 28) * 2 + np.sin(2 * np.pi * 85 * tt) * np.exp(-tt * 20))
        tt = t(0.3)
        fx["squish"] = self._snd(low(noise(0.3), 30) * np.exp(-tt * 10) * 3 + sweep(220, 50, 0.3) * np.exp(-tt * 9) * 0.6)
        tt = t(0.06)
        fx["flick"] = self._snd(noise(0.06) * np.exp(-tt * 90))
        tt = t(1.0)
        fx["boom"] = self._snd(low(noise(1.0), 60) * np.exp(-tt * 4) * 5 + np.sin(2 * np.pi * 45 * tt) * np.exp(-tt * 5), 0.8)
        tt = t(0.4)
        crackle = noise(0.4) * (rng.random(len(tt)) < 0.25)
        fx["zap"] = self._snd((np.sign(np.sin(2 * np.pi * 70 * tt)) * 0.4 + crackle) * np.exp(-tt * 6), 0.5)
        fx["sizzle"] = self._snd((noise(1.0) - low(noise(1.0), 4)) * (0.5 + 0.5 * (rng.random(R) < 0.08)), 0.35)
        fx["hiss"] = self._snd(noise(1.0) - low(noise(1.0), 3), 0.25)
        tt = t(0.2)
        bite = low(noise(0.2), 8) * 2.5 * (np.exp(-tt * 40) + np.exp(-np.maximum(tt - 0.1, 0) * 40) * (tt > 0.1))
        fx["chomp"] = self._snd(bite)
        fx["yum"] = self._snd(np.concatenate([np.sin(2 * np.pi * 520 * t(0.09)) * np.exp(-t(0.09) * 20),
                                              np.sin(2 * np.pi * 780 * t(0.16)) * np.exp(-t(0.16) * 12)]), 0.4)
        tt = t(0.6)
        pings = sum(np.sin(2 * np.pi * f * tt) * np.exp(-tt * 9) for f in (2200, 2950, 3700, 4400))
        fx["shatter"] = self._snd((noise(0.6) - low(noise(0.6), 3)) * np.exp(-tt * 7) + 0.25 * pings, 0.5)
        tt = t(0.6)
        fx["splash"] = self._snd(low(noise(0.6), 12) * np.exp(-tt * 6) * 2 + 0.3 * sweep(300, 900, 0.6) * np.exp(-tt * 10))
        tt = t(1.0)
        fx["whoosh"] = self._snd(low(noise(1.0), 40) * 4 * (0.7 + 0.3 * np.sin(2 * np.pi * tt)), 0.35)
        tt = t(0.9)
        fx["death"] = self._snd(sweep(440, 90, 0.9) * np.exp(-tt * 2.5), 0.45)
        tt = t(0.18)
        fx["dodge"] = self._snd(sweep(180, 950, 0.18) * np.exp(-tt * 6), 0.45)
        tt = t(0.07)
        fx["pop"] = self._snd(np.sin(2 * np.pi * 900 * tt) * np.exp(-tt * 50), 0.4)
        tt = t(0.14)
        fx["bonk"] = self._snd(np.sin(2 * np.pi * 150 * tt) * np.exp(-tt * 25) + noise(0.14) * np.exp(-tt * 120) * 0.5)
        tt = t(0.3)
        fx["drop"] = self._snd(sweep(1400, 500, 0.3) * np.exp(-tt * 5), 0.25)
        tt = t(0.5)
        fx["hum"] = self._snd(np.sin(2 * np.pi * 120 * tt) * 0.5 + np.sin(2 * np.pi * 240 * tt) * 0.2, 0.2)
        tt = t(0.25)
        fx["click"] = self._snd(np.sin(2 * np.pi * 1300 * tt) * np.exp(-tt * 40), 0.3)
        return fx

    def play(self, name: str, vol: float = 1.0) -> None:
        if self.ok and not self.muted and name in self.fx:
            ch = self.fx[name].play()
            if ch:
                ch.set_volume(vol)

    def loop(self, slot: str, name: str | None, vol: float = 1.0) -> None:
        """Keep one looping sound per slot; None (or muted) stops it."""
        if not self.ok:
            return
        cur = self.loops.get(slot)
        if name is None or self.muted:
            if cur:
                cur[1].stop()
                del self.loops[slot]
            return
        if cur and cur[0] == name:
            cur[1].set_volume(vol)
            return
        if cur:
            cur[1].stop()
        ch = self.fx[name].play(loops=-1)
        if ch:
            ch.set_volume(vol)
            self.loops[slot] = (name, ch)


SURGERY = (  # label, how to find the neurons (see Game._surgery_rows)
    ("Giant fiber escape neuron (DNp01)", ("type", ("DNp01",))),
    ("Looming detectors (LPLC2, LC4)", ("group", ("loom",))),
    ("Touch neurons", ("group", TOUCH)),
    ("Pain relay: ascending neurons", ("pop", ("ascending",))),
    ("Heat and cold sensors", ("group", ("heat", "cold"))),
    ("Smell neurons (ORNs)", ("group", ("smell",))),
    ("Mushroom bodies: Kenyon cells", ("prefix", ("KC",))),
    ("Reward dopamine neurons (PAM)", ("group", ("reward",))),
    ("Punishment dopamine neurons (PPL1)", ("group", ("punish",))),
    ("Moonwalker neurons (MDN)", ("group", ("back",))),
    ("Walking neurons (DNp09)", ("group", ("walk",))),
    ("Flight power neurons (DNg02)", ("group", ("fly",))),
    ("Central complex (compass, steering)", ("prefix", ("EPG", "PEN", "PEG", "PFL", "PFN", "PFG", "PFR", "ER", "FB",
                                                         "hDelta", "vDelta", "FC", "FS", "EL", "ExR"))),
    ("Optic lobes and photoreceptors", ("pop", ("optic lobe", "photoreceptors"))),
    ("Every neuron", ("all", ())),
)
HELP = (
    ("1-9, 0", "pick a tool (or click the toolbar)"),
    ("B", "big live brain view; click a neuron to inspect it"),
    ("O", "brain surgery: silence or stimulate neuron groups"),
    ("E", "change arena: room, fan, flypaper, pool, lamp"),
    ("P", "pain neurons: normal, more, max"),
    ("I", "immortal mode"),
    ("M", "mute sound"),
    ("S / G", "save a screenshot / a GIF of the last 6 seconds"),
    ("R", "new fly"),
    ("F11", "fullscreen (or Alt+Enter); drag the window edge to resize"),
    ("H", "this help"),
    ("Esc", "quit"),
)


class Game:
    def __init__(self, screen, brain: Brain, view: BrainView):
        self.screen, self.brain = screen, brain
        self.tool = 0
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
        self.born_view = time.perf_counter()
        self.immortal = False
        self.pain_level = 0
        self.sound = Sound()
        self.arena_i = 0
        self.surgery_open = self.help_open = False
        self.surgery_rows = {label: self._surgery_rows(spec) for label, spec in SURGERY}
        self.surgery_modes = [0] * len(SURGERY)
        self.type_ops: dict[str, int] = {}
        self.surgery_buttons: list = []
        self.inspect: dict | None = None
        self.inspect_buttons: list = []
        self.big_rect = pygame.Rect(0, 0, 0, 0)
        self.frames: deque = deque(maxlen=GIF_FRAMES)
        self.frame = 0
        self.saved_msg: tuple[str, float] | None = None
        self.mouse = (0, 0)
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
        self.shards: list[list] = []
        self.bolts: list[list] = []
        self.spider: dict | None = None
        self.sugars: list[dict] = []
        self.zap_ready = 0.0
        self.brain.sedation = 0.0
        self.reward = 0.0
        self.pain = 0.0
        self.pain_parts = np.zeros(5)
        self.last_damage = 0.0
        self.pain_peak = 0.0
        self.pain_max_s = 0.0
        self.pain_trace: list[float] = []
        br = self.brain
        self.fear_w = np.zeros(len(br.kc), np.float32)   # mushroom body memory, per Kenyon cell
        self.like_w = np.zeros(len(br.kc), np.float32)
        self.kc_calm = br.sim.activity.rates()[br.kc].copy()
        self.templates: dict[str, np.ndarray] = {}
        self.fear_now = self.like_now = 0.0
        self.avoid_ready = 0.0
        self.loom, self.loom_prev, self.threat_x = 0.0, {}, 0.0
        self.scent_now: str | None = None
        self.sugar_scent = False
        self.streaks: list[list] = []
        self.photo_ready = 0.0
        self.death_frames: list = []
        self.surgery_modes = [0] * len(SURGERY)
        self.type_ops = {}
        br.clear_overrides()
        if br.dead:
            br.revive()

    def _view_loop(self) -> None:
        """Renders the brain view at ~20 Hz on its own thread (10-25 ms per render; the sparse math releases the GIL)."""
        while not self.view_stop:
            t0 = time.perf_counter()
            key = "big" if self.big_view else "panel"
            br = self.brain
            calm = not br.dead and br.sedation == 0 and br.steps - br.last_poke > CALM_STEPS
            raster = br.sim.activity.raster()
            spiked = raster[-1] if raster else np.zeros(0, np.int64)
            self.view_surf[key] = self.view.render(key, br.sim.activity.rates(), spiked, t0 - self.born_view, learn=calm)
            time.sleep(max(0.005, 0.05 - (time.perf_counter() - t0)))

    def _view_surface(self, key: str) -> pygame.Surface:
        surf = self.view_surf.get(key)
        if surf is None:
            surf = pygame.Surface(VIEW_SIZES[key])
            surf.fill((4, 5, 8))
        return surf

    def _hud_overlay(self, surf, rect: pygame.Rect, now: float, small: bool) -> None:
        """Sci-fi dressing over a brain view: a sweeping scan band, corner brackets and live counters."""
        ph = (now * 0.22) % 1.0
        band_y = rect.y + int(ph * rect.h)
        band = pygame.Surface((rect.w, 34))
        for k in range(34):                          # additive glow that fades in toward the scan line
            a = 0.16 * (k / 33) ** 2
            band.fill((int(80 * a), int(200 * a), int(255 * a)), (0, k, rect.w, 1))
        clip = surf.get_clip()
        surf.set_clip(rect)
        surf.blit(band, (rect.x, band_y - 34), special_flags=pygame.BLEND_RGB_ADD)
        pygame.draw.line(surf, (60, 120, 150), (rect.x, band_y), (rect.right, band_y), 1)
        surf.set_clip(clip)
        L = 10 if small else 22
        for cx, cy, sx, sy in ((rect.x, rect.y, 1, 1), (rect.right - 1, rect.y, -1, 1),
                               (rect.x, rect.bottom - 1, 1, -1), (rect.right - 1, rect.bottom - 1, -1, -1)):
            pygame.draw.line(surf, ACCENT, (cx, cy), (cx + sx * L, cy), 2)
            pygame.draw.line(surf, ACCENT, (cx, cy), (cx, cy + sy * L), 2)
        v = self.view
        if small:
            self._text(surf, f"{v.firing:,} firing", (rect.x + 6, rect.y + 4), (150, 215, 240), self.f_small)
            self._text(surf, f"{v.hot_firing:,} pain", (rect.right - 6, rect.y + 4), (255, 150, 70), self.f_small, "topright")

    def _draw_big_view(self, surf) -> None:
        w, h = VIEW_SIZES["big"]
        now = time.perf_counter()
        veil = pygame.Surface((PLAY_W, h + 120), pygame.SRCALPHA)
        pygame.draw.rect(veil, (3, 4, 7, 252), veil.get_rect(), border_radius=14)
        surf.blit(veil, (0, 0))
        rect = pygame.Rect((PLAY_W - w) // 2, 58, w, h)
        self.big_rect = rect
        surf.blit(self._view_surface("big"), rect)
        self._hud_overlay(surf, rect, now, small=False)
        if self.inspect is not None:
            self._draw_inspect(surf, rect, now)
        labels = (("optic lobe", 0.10, 0.18), ("optic lobe", 0.90, 0.18), ("mushroom bodies", 0.50, 0.06),
                  ("central brain", 0.50, 0.42), ("to nerve cord", 0.50, 0.93))
        for label, fx, fy in labels if self.inspect is None else ():
            self._text(surf, label.upper(), (rect.x + int(fx * w), rect.y + int(fy * h)), (120, 170, 190), self.f_small, "center")
        self._text(surf, "LIVE CONNECTOME", (22, 16), INK, self.f_head)
        v = self.view
        pulse = 0.5 + 0.5 * math.sin(now * 6)
        self._text(surf, f"{v.firing:,} neurons firing above normal", (230, 20), (150, 215, 240), self.f_bold)
        self._text(surf, f"{v.hot_firing:,} pain neurons", (520, 20), tuple(int(c * (0.7 + 0.3 * pulse)) for c in (255, 150, 70)), self.f_bold)
        self._text(surf, "click a neuron to inspect   |   B to close", (PLAY_W - 22, 42), LABEL, self.f_small, "topright")
        ly = rect.bottom + 12
        aacircle(surf, (30, ly + 7), 5, (255, 130, 40))
        r = self._text(surf, "pain-sensing neurons firing", (42, ly), TEXT, self.f_small)
        aacircle(surf, (r.right + 22, ly + 7), 5, (110, 200, 255))
        r = self._text(surf, "other neurons firing", (r.right + 34, ly), TEXT, self.f_small)
        self._text(surf, "dim wiring colored by fiber direction, front brighter", (r.right + 22, ly), LABEL, self.f_small)
        self._text(surf, "Fibers run from each neuron's real cell body toward its synaptic partners (estimated shapes).",
                   (24, ly + 20), DIM, self.f_small)

    # --- seeing, smelling, learning ---------------------------------------------
    def _overlay_open(self) -> bool:
        return self.report is not None or self.big_view or self.surgery_open or self.help_open

    def _threats(self, now: float, mouse) -> list:
        out = []
        name = TOOLS[self.tool][0]
        if not self._overlay_open() and mouse[0] < PLAY_W and mouse[1] < FLOOR and name not in ("hand", "sugar"):
            out.append(("cursor", np.array(mouse, float), CURSOR_SIZE.get(name, 16)))
        for sw in self.swats:
            ph = (now - sw[1]) / 0.55
            if ph < 0.3:                              # the swatter head on its way down (same path as draw_swatter)
                pos = np.array(sw[0], float)
                pivot = pos + (260.0, 330.0)
                d = pos - pivot
                ang = math.atan2(d[1], d[0]) - 0.9 * (1 - (ph / 0.3) ** 2)
                out.append((("swat", id(sw)), pivot + np.array([math.cos(ang), math.sin(ang)]) * float(np.hypot(*d)), 80.0))
        if self.spider is not None and self.spider["state"] == "hunt":
            out.append(("spider", self.spider["p"].copy(), 22.0))
        for b in self.bombs:
            out.append((("bomb", id(b)), b["p"].copy(), 16.0))
        return out

    def _vision(self, now: float, mouse) -> None:
        """Looming: how fast each object's angular size grows as seen from the fly's head. The game computes this and
        drives the real looming detectors LPLC2/LC4, which excite the giant fiber in the connectome (see docstring)."""
        fly = self.fly
        if fly.dead or fly.frozen_at is not None:
            self.loom = 0.0
            return
        head = fly.p[HEAD]
        best, best_pos, seen = 0.0, None, {}
        for key, pos, r in self._threats(now, mouse):
            d = max(float(np.hypot(*(pos - head))), r + 4.0)
            theta = 2 * math.atan(r / d)
            prev = self.loom_prev.get(key)
            seen[key] = theta
            if prev is not None and (theta - prev) * 60.0 > best:
                best, best_pos = (theta - prev) * 60.0, pos
        self.loom_prev = seen
        self.loom += (best - self.loom) * 0.5
        strength = float(np.clip((best - LOOM_MIN) / LOOM_FULL, 0, 1))
        if strength > 0 and best_pos is not None:
            self.brain.poke("loom", None, strength, recruit=0.6 * strength)
            self.threat_x = float(best_pos[0])

    def _scents(self, now: float, mouse) -> None:
        """Each tool carries its own scent (6 of the 53 olfactory glomeruli; a game rule) that the fly smells up close."""
        fly = self.fly
        self.scent_now, self.sugar_scent = None, False
        if fly.dead:
            return
        name = TOOLS[self.tool][0]
        near = np.hypot(*(np.array(mouse, float) - fly.p[HEAD])) < SCENT_RANGE
        if not self._overlay_open() and mouse[0] < PLAY_W and near:
            self.scent_now = name
            self.brain.poke("scent", name, 0.3)
        if any(abs(sg["p"][0] - fly.p[HEAD, 0]) < 400 for sg in self.sugars):
            self.sugar_scent = True
            self.brain.poke("scent", "sugar", 0.3)

    def _learn(self, now: float) -> None:
        """Mushroom body learning, a rule added on top of the fixed connectome. Kenyon cells active with a scent get
        their fear weight raised while the PPL1 punishment neurons fire, and their liking raised while PAM reward
        neurons fire. Recall is the current Kenyon cell pattern read through those weights."""
        br, fly = self.brain, self.fly
        if fly.dead:
            return
        r = br.sim.activity.rates()[br.kc]
        present = self.scent_now is not None or self.sugar_scent
        if not present and br.steps - br.last_poke > CALM_STEPS:
            self.kc_calm += (r - self.kc_calm) * 0.02
        act = np.maximum(r - self.kc_calm - 0.01, 0)
        if np.count_nonzero(act) > KC_SPARSE:          # Kenyon cells code sparsely: keep the most active ~7%
            act[act < np.partition(act, -KC_SPARSE)[-KC_SPARSE]] = 0
        total = float(act.sum())
        if not present or total <= 0:
            self.fear_now = self.like_now = 0.0
            return
        a = (act / total).astype(np.float32)
        self.fear_now, self.like_now = float(a @ self.fear_w), float(a @ self.like_w)
        g = np.minimum(a * np.count_nonzero(act), 3.0)
        punish = float(np.clip((br.level("punish") - 1.3) / 1.2, 0, 1))
        reward = float(np.clip((br.level("reward") - 1.3) / 1.0, 0, 1))
        if punish > 0:
            self.fear_w += LEARN_RATE * punish * g * (1 - self.fear_w)
        if reward > 0:
            self.like_w += LEARN_RATE * reward * g * (1 - self.like_w)
        self.fear_w *= np.float32(0.99998)            # slow forgetting
        self.like_w *= np.float32(0.99998)
        for key in ([self.scent_now] if self.scent_now else []) + (["sugar"] if self.sugar_scent else []):
            t = self.templates.get(key)
            self.templates[key] = a.copy() if t is None else t + (a - t) * 0.05

    def memory_of(self, key: str) -> tuple[float, float]:
        t = self.templates.get(key)
        return (0.0, 0.0) if t is None else (float(t @ self.fear_w), float(t @ self.like_w))

    def _memory_behavior(self, now: float, free: bool, can_fly: bool) -> None:
        fly = self.fly
        if not self.scent_now or not free or now < self.avoid_ready or fly.frozen_at is not None or now < fly.stun_until:
            return
        mx = self.mouse[0]
        if self.scent_now != "sugar" and self.fear_now > FEAR_ACT:
            self.avoid_ready = now + 2.5
            fly.last_hit_x = mx
            if can_fly and self.fear_now > 0.6:
                fly.escape(now)
            else:
                fly.facing = 1 if fly.p[THX, 0] >= mx else -1
                fly.walk_until, fly.back_until, fly.run = now + 1.2, 0.0, True
            self.note(f"AVOID    remembers the {self.scent_now} ({self.fear_now:.2f})")
            self.popup(fly.p[HEAD] + (0, -60), "NOPE!", (255, 220, 120))
        elif self.scent_now == "sugar" and self.like_now > LIKE_ACT and now >= fly.walk_until:
            self.avoid_ready = now + 1.5
            fly.facing = 1 if mx > fly.p[THX, 0] else -1
            fly.walk_until, fly.run = now + 1.0, False
            self.note(f"APPROACH remembers sugar ({self.like_now:.2f})")

    # --- arenas --------------------------------------------------------------------
    def _environment(self, now: float, mouse) -> None:
        fly, br = self.fly, self.brain
        arena = ARENAS[self.arena_i]
        fly.arena, fly.wind = arena, 0.0
        if arena != "flypaper":
            fly.stuck.clear()
        if arena == "fan":
            gust = 0.75 + 0.25 * math.sin(now * 1.3) + 0.15 * math.sin(now * 4.1)
            fly.wind = 0.55 * gust * float(np.clip(1.15 - fly.p[THX, 0] / 900.0, 0.25, 1.0))
            if not fly.dead and self.frame % 3 == 0:
                br.poke("wind", None, min(1.0, fly.wind * 1.4))     # Johnston's organ wind neurons
            if random.random() < 0.7:
                self.streaks.append([FAN[0] + 50, FAN[1] + random.uniform(-80, 80), random.uniform(10, 18), now])
        elif arena == "flypaper" and fly.stuck:
            kicking = now < fly.flail_until
            for i in list(fly.stuck):
                pull = fly.grabbed is not None and np.hypot(*(np.array(mouse, float) - fly.stuck[i])) > 90
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
            sub = fly.p[:, 1] > WATER_Y
            if sub.any():
                if fly.wet <= 0 and float(np.max((fly.p - fly.prev)[sub, 1])) > 4:
                    self.sound.play("splash")
                    self.puff((fly.p[THX, 0], WATER_Y), 10, 3)
                fly.wet = 3.0
                if not fly.dead:
                    if self.frame % 4 == 0:
                        br.poke("humid", None, 0.9)             # hygrosensory neurons
                        br.poke("body", None, 0.25)
                    if fly.p[HEAD, 1] > WATER_Y + 6:
                        self.damage(0.05, "drowning")
        elif arena == "lamp":
            d = float(np.hypot(*(fly.p[HEAD] - LAMP)))
            if not fly.dead:
                light = float(np.clip(1.2 - d / 500.0, 0.15, 1.0))
                if self.frame % 2 == 0:
                    br.poke("light", None, light, recruit=0.25 * light)   # photoreceptors
                if d < 58:
                    br.poke("heat", None, 0.8)
                    self.damage(0.08, "the hot lamp")
                    away = (fly.p[HEAD] - LAMP) / max(d, 1.0)
                    for i in (HEAD, THX, ABD):
                        fly.impulse(i, away * 1.5)
            if now < fly.escape_until and np.hypot(*(fly.fly_target - LAMP)) > 130:
                fly.fly_target = np.array(LAMP) + (random.uniform(-110, 110), random.uniform(50, 130))
        if arena != "pool":
            fly.wet = max(0.0, fly.wet - 1 / 60)
        elif not (fly.p[:, 1] > WATER_Y).any():
            fly.wet = max(0.0, fly.wet - 1 / 60)
        self.streaks = [st for st in self.streaks if now - st[3] < 1.2]

    def _draw_arena_back(self, surf, now: float) -> None:
        arena = ARENAS[self.arena_i]
        if arena == "fan":
            cx, cy = FAN
            thick_line(surf, (cx - 20, cy), (cx - 20, FLOOR), 10, (70, 74, 84))
            aacircle(surf, (cx, cy), 78, (50, 54, 64))
            aacircle(surf, (cx, cy), 70, (26, 28, 34))
            for k in range(3):
                a = now * 25 + k * 2.094
                tip = (cx + 62 * math.cos(a), cy + 62 * math.sin(a))
                aapoly(surf, ellipse_pts(((cx + tip[0]) / 2, (cy + tip[1]) / 2), 34, 13, a, 16), (150, 160, 175))
            aacircle(surf, (cx, cy), 12, (90, 96, 108))
            for k in range(-3, 4):
                gfxdraw.line(surf, int(cx - 70), int(cy + k * 18), int(cx + 70), int(cy + k * 18), (95, 100, 112))
            for st in self.streaks:
                e = (now - st[3]) / 1.2
                x = st[0] + e * 1100 * (st[2] / 14)
                gfxdraw.line(surf, int(x), int(st[1]), int(x + 40), int(st[1]), (200, 220, 235, int(110 * (1 - e))))
        elif arena == "flypaper":
            x0, x1 = PAPER_X
            pygame.draw.rect(surf, (214, 176, 50), (x0, FLOOR - 5, x1 - x0, 9), border_radius=3)
            pygame.draw.rect(surf, (245, 214, 95), (x0 + 6, FLOOR - 4, x1 - x0 - 12, 2))
            for k in range(12):
                gfxdraw.filled_circle(surf, int(x0 + 20 + k * 35), FLOOR + 1, 2, (170, 130, 40))
            self._text(surf, "FLYPAPER", ((x0 + x1) // 2, FLOOR + 10), (190, 160, 70), self.f_small, "midtop")
        elif arena == "lamp":
            lx, ly = LAMP
            pygame.draw.line(surf, (60, 60, 64), (lx, CEIL - 40), (lx, ly - 40), 3)
            aapoly(surf, [(lx - 22, ly - 42), (lx + 22, ly - 42), (lx + 46, ly - 8), (lx - 46, ly - 8)], (60, 64, 74))
            pulse = 0.9 + 0.1 * math.sin(now * 3)
            for r, a in ((190, 18), (120, 30), (70, 60)):
                aacircle(surf, (lx, ly), r * pulse, (255, 230, 150, a))
            aacircle(surf, (lx, ly), 24, (255, 246, 205))

    def _draw_arena_front(self, surf, now: float) -> None:
        if ARENAS[self.arena_i] != "pool":
            return
        water = pygame.Surface((PLAY_W, FLOOR - WATER_Y), pygame.SRCALPHA)
        water.fill((40, 120, 190, 95))
        surf.blit(water, (0, WATER_Y))
        pts = [(x, WATER_Y + 3 * math.sin(x * 0.03 + now * 2.2)) for x in range(0, PLAY_W + 20, 20)]
        pygame.draw.aalines(surf, (170, 220, 255), False, pts)
        for k in range(6):
            x = (k * 157 + now * 30) % PLAY_W
            gfxdraw.line(surf, int(x), WATER_Y + 20 + k * 17, int(x + 50), WATER_Y + 20 + k * 17, (150, 210, 250, 70))

    # --- brain surgery ---------------------------------------------------------------
    def _surgery_rows(self, spec) -> np.ndarray:
        kind, names = spec
        br = self.brain
        if kind == "group":
            return np.flatnonzero(np.isin(br.det_id, [br.col[n] for n in names]))
        if kind == "pop":
            pops = [n for n, _ in POPS]
            return np.flatnonzero(np.isin(br.pop_id, [pops.index(n) for n in names]))
        if kind == "type":
            return np.flatnonzero(np.isin(br.types, names))
        if kind == "prefix":
            m = np.zeros(br.n, bool)
            for pre in names:
                m |= np.char.startswith(br.types, pre)
            return np.flatnonzero(m)
        return np.arange(br.n)

    def _apply_surgery(self) -> None:
        br = self.brain
        br.clear_overrides()
        for (label, _), mode in zip(SURGERY, self.surgery_modes):
            if mode:
                br.set_override(self.surgery_rows[label], mode)
        for t, mode in self.type_ops.items():
            if mode:
                br.set_override(np.flatnonzero(br.types == t), mode)

    def _set_surgery(self, k: int, mode: int) -> None:
        self.surgery_modes[k] = mode
        self._apply_surgery()
        label = SURGERY[k][0]
        self.note(f"SURGERY  {label.split(' (')[0]}: {'off' if mode < 0 else 'on' if mode > 0 else 'normal'}")
        self.sound.play("click")

    def _draw_surgery(self, surf) -> None:
        panel = pygame.Rect(95, 36, 700, 590)
        veil = pygame.Surface((PLAY_W, H), pygame.SRCALPHA)
        veil.fill((4, 5, 8, 170))
        surf.blit(veil, (0, 0))
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, BORDER, panel, 1, border_radius=16)
        x = panel.x + 24
        self._text(surf, "BRAIN SURGERY", (x, panel.y + 16), INK, self.f_title)
        self._text(surf, "Silence (OFF) or stimulate (ON) real neuron groups, then close with O and watch the fly and its brain.",
                   (x + 2, panel.y + 58), LABEL, self.f_small)
        y = panel.y + 88
        self.surgery_buttons = []
        for k, (label, _) in enumerate(SURGERY):
            n = len(self.surgery_rows[label])
            mode = self.surgery_modes[k]
            self._text(surf, label, (x, y), AMBER if mode else TEXT, self.f_text)
            self._text(surf, f"{n:,}", (panel.right - 250, y + 2), LABEL, self.f_small, "topright")
            for j, (m, text) in enumerate(((-1, "OFF"), (0, "-"), (1, "ON"))):
                r = pygame.Rect(panel.right - 226 + j * 66, y - 1, 58, 24)
                on = mode == m
                fill = ((60, 110, 200) if m < 0 else (230, 130, 50) if m > 0 else (70, 76, 90)) if on else (30, 34, 44)
                pygame.draw.rect(surf, fill, r, border_radius=6)
                self._text(surf, text, r.center, INK if on else LABEL, self.f_small, "center")
                self.surgery_buttons.append((r, k, m))
            y += 30
        if self.type_ops:
            ops = ", ".join(f"{t} {'off' if m < 0 else 'on'}" for t, m in self.type_ops.items() if m)
            self._text(surf, f"From the inspector: {ops}", (x, y + 4), AMBER, self.f_small)
        clear = pygame.Rect(panel.right - 160, panel.bottom - 44, 136, 30)
        pygame.draw.rect(surf, (60, 64, 76), clear, border_radius=8)
        self._text(surf, "CLEAR ALL", clear.center, INK, self.f_bold, "center")
        self.surgery_buttons.append((clear, -1, 0))
        self._text(surf, "OFF adds -2.4 per step (silences them). ON adds +0.48 (fires them fast).", (x, panel.bottom - 36), DIM, self.f_small)

    # --- neuron inspector ----------------------------------------------------------------
    def _inspect_at(self, pos) -> None:
        w, h = VIEW_SIZES["big"]
        px, py = pos[0] - self.big_rect.x, pos[1] - self.big_rect.y
        pix = self.view.spark_pix["big"]
        cx, cy = pix % w, pix // w
        near = np.flatnonzero((pix >= 0) & (np.abs(cx - px) <= 7) & (np.abs(cy - py) <= 7))
        if not len(near):
            self.inspect = None
            return
        rates = self.brain.sim.activity.rates()[near]
        dist = np.hypot(cx[near] - px, cy[near] - py)
        self.inspect = self._neuron_info(int(near[np.argmax(rates / 0.025 * 0.5 - dist)]))   # nearest, then busiest
        self.sound.play("click")

    def _neuron_info(self, i: int) -> dict:
        br = self.brain
        Wr, Wc = br.sim.W_csr, br.sim.W_csc

        def top(idx, val):
            order = np.argsort(-np.abs(val))[:6]
            return [(int(idx[k]), float(val[k])) for k in order]

        ins = top(Wr.indices[Wr.indptr[i]:Wr.indptr[i + 1]], Wr.data[Wr.indptr[i]:Wr.indptr[i + 1]])
        outs = top(Wc.indices[Wc.indptr[i]:Wc.indptr[i + 1]], Wc.data[Wc.indptr[i]:Wc.indptr[i + 1]])
        pop = br.pop_id[i]
        return {"i": i, "type": br.types[i] or "untyped", "instance": br.instance[i],
                "pop": [n for n, _ in POPS][pop] if pop >= 0 else br.superclass[i], "ins": ins, "outs": outs,
                "n_in": int(Wr.indptr[i + 1] - Wr.indptr[i]), "n_out": int(Wc.indptr[i + 1] - Wc.indptr[i])}

    def _draw_inspect(self, surf, rect: pygame.Rect, now: float) -> None:
        info = self.inspect
        br, w = self.brain, VIEW_SIZES["big"][0]
        pix = self.view.spark_pix["big"]

        def at(j):
            q = pix[j]
            return None if q < 0 else (rect.x + q % w, rect.y + q // w)

        me = at(info["i"])
        for lst, col in ((info["ins"], (110, 200, 255)), (info["outs"], (255, 160, 80))):
            for j, _ in lst:
                q = at(j)
                if me and q:
                    pygame.draw.aaline(surf, col, me, q)
                    aacircle(surf, q, 3, col)
        if me:
            r = 8 + 3 * math.sin(now * 8)
            gfxdraw.aacircle(surf, int(me[0]), int(me[1]), int(r), (255, 255, 255))
            gfxdraw.aacircle(surf, int(me[0]), int(me[1]), int(r + 4), (255, 255, 255, 120))
        card = pygame.Rect(rect.right - 318, rect.y + 10, 308, 300)
        bg = pygame.Surface(card.size, pygame.SRCALPHA)
        pygame.draw.rect(bg, (8, 10, 16, 225), bg.get_rect(), border_radius=10)
        surf.blit(bg, card)
        x, y = card.x + 12, card.y + 10
        i = info["i"]
        rate = float(br.sim.activity.rates()[i]) / 0.005
        calm = float(self.view.calm[i]) / 0.005
        self._text(surf, info["type"], (x, y), INK, self.f_head)
        self._text(surf, f"{info['pop']}   {info['instance']}", (x, y + 26), LABEL, self.f_small)
        self._text(surf, f"firing {rate:5.1f} spikes/s   (calm {calm:4.1f})", (x, y + 44),
                   (255, 170, 90) if rate > calm + 2 else TEXT, self.f_small)
        self._text(surf, f"{info['n_in']} input partners, {info['n_out']} output partners", (x, y + 60), LABEL, self.f_small)
        yy = y + 80
        for title, lst, col in (("strongest inputs (% of its input)", info["ins"], (110, 200, 255)),
                                ("strongest outputs (% of target's input)", info["outs"], (255, 160, 80))):
            self._text(surf, title, (x, yy), col, self.f_small)
            yy += 15
            for j, v in lst[:4]:
                name = br.types[j] or "untyped"
                side = br.instance[j][-2:] if br.instance[j].endswith(("_L", "_R")) else ""
                sign = "+" if v > 0 else "-"
                self._text(surf, f"{sign}{abs(v) * 100:4.1f}%  {name}{side}", (x + 6, yy), TEXT, self.f_small)
                yy += 14
            yy += 4
        self.inspect_buttons = []
        t = info["type"]
        for k, (label, mode) in enumerate((("SILENCE TYPE", -1), ("STIMULATE", 1), ("CLEAR", 0))):
            r = pygame.Rect(card.x + 12 + k * 98, card.bottom - 34, 90, 24)
            active = self.type_ops.get(t, 0) == mode and mode != 0
            pygame.draw.rect(surf, (60, 110, 200) if (active and mode < 0) else (230, 130, 50) if active else (40, 46, 58), r, border_radius=6)
            self._text(surf, label, r.center, INK, self.f_small, "center")
            self.inspect_buttons.append((r, mode))

    # --- sound, saving, help ---------------------------------------------------------------
    def _sound_update(self, now: float) -> None:
        snd, fly = self.sound, self.fly
        flying = fly.flying and now < fly.escape_until
        snd.loop("wings", f"buzz{min(3, max(0, int((fly.power - 0.6) * 3)))}" if flying else None, 0.55)
        name = TOOLS[self.tool][0]
        tool = None
        if self.torching and self.report is None:
            tool = "sizzle" if name == "torch" else "hiss" if name in ("cleaner", "freeze") else None
        snd.loop("tool", tool, 0.6)
        arena = ARENAS[self.arena_i]
        snd.loop("arena", "whoosh" if arena == "fan" else "hum" if arena == "lamp" else None, 0.5)

    def capture(self) -> None:
        small = pygame.transform.smoothscale(self.screen, GIF_SIZE)
        self.frames.append(pygame.image.tobytes(small, "RGB"))

    def _save_dir(self) -> Path:
        for d in (Path.home() / "Pictures" / "Kick the Fly", Path.cwd() / "Kick the Fly saves"):
            try:
                d.mkdir(parents=True, exist_ok=True)
                return d
            except OSError:
                continue
        return Path.cwd()

    def save_png(self) -> None:
        path = self._save_dir() / f"kick-the-fly-{time.strftime('%Y%m%d-%H%M%S')}.png"
        pygame.image.save(self.screen, str(path))
        self.saved_msg = (f"saved {path}", time.perf_counter())
        self.sound.play("click")

    def save_gif(self, frames: list | None = None) -> None:
        frames = list(self.frames) if frames is None else list(frames)
        if len(frames) < 3:
            self.saved_msg = ("not enough footage yet", time.perf_counter())
            return
        path = self._save_dir() / f"kick-the-fly-{time.strftime('%Y%m%d-%H%M%S')}.gif"
        self.saved_msg = ("saving GIF...", time.perf_counter())

        def work():
            try:
                from PIL import Image

                imgs = [Image.frombytes("RGB", GIF_SIZE, f).convert("P", palette=Image.ADAPTIVE, colors=160) for f in frames]
                imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=66, loop=0)
                self.saved_msg = (f"saved {path}", time.perf_counter())
            except Exception as e:                   # e.g. Pillow missing in a source checkout
                self.saved_msg = (f"GIF failed: {e}", time.perf_counter())

        threading.Thread(target=work, daemon=True).start()
        self.sound.play("click")

    def _draw_help(self, surf) -> None:
        panel = pygame.Rect(165, 110, 560, 64 + 30 * len(HELP))
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, BORDER, panel, 1, border_radius=16)
        self._text(surf, "CONTROLS", (panel.x + 24, panel.y + 16), INK, self.f_head)
        for k, (key, what) in enumerate(HELP):
            y = panel.y + 54 + k * 30
            self._text(surf, key, (panel.x + 30, y), AMBER, self.f_bold)
            self._text(surf, what, (panel.x + 120, y), TEXT, self.f_text)

    def _draw_memory(self, surf) -> None:
        learned = [(k, *self.memory_of(k)) for k in self.templates]
        x, y, w = 10, 380, 236
        rows = sorted((r for r in learned if max(r[1], r[2]) >= 0.02), key=lambda r: -max(r[1], r[2]))[:4]
        h = 46 + 16 * max(1, len(rows))
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 180), card.get_rect(), border_radius=10)
        surf.blit(card, (x, y))
        self._text(surf, "MEMORY", (x + 12, y + 8), LABEL, self.f_small)
        self._text(surf, "mushroom body", (x + w - 12, y + 8), DIM, self.f_small, "topright")
        yy = y + 28
        if not rows:
            self._text(surf, "nothing learned yet", (x + 12, yy), DIM, self.f_small)
        for key, fear, like in rows:
            good = like > fear
            v = like if good else fear
            self._text(surf, f"{key}", (x + 12, yy - 3), TEXT, self.f_small)
            self._bar(surf, x + 84, yy, w - 150, v, (255, 150, 190) if good else (240, 150, 60))
            self._text(surf, f"{'likes' if good else 'fears'} {v:.2f}", (x + w - 12, yy - 3), TEXT, self.f_small, "topright")
            yy += 16
        self._text(surf, "learning rule added to the connectome", (x + 12, y + h - 16), DIM, self.f_small)

    def _above_head(self):
        """Where a popup over the fly goes (the 3D game overrides this)."""
        return self.fly.p[HEAD] + (0, -70)

    def _panel_image(self, surf: pygame.Surface) -> pygame.Surface:
        """In see-through panel modes the brain image's black background turns transparent; bright neurons stay."""
        pa = getattr(self, "panel_alpha", 255)
        if pa >= 255:
            return surf
        out = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        out.blit(surf, (0, 0))
        rgb = pygame.surfarray.pixels3d(out)
        alpha = pygame.surfarray.pixels_alpha(out)
        alpha[:] = np.clip(rgb.max(axis=2).astype(np.int16) * 2 + pa // 3, 0, 255)
        del rgb, alpha
        return out

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
        if fly.frozen_at is not None and fly.shattered_at is None and name in ("hand", "flick", "swatter", "zapper"):
            if fly.nearest(pos, 60) is not None:
                self._shatter(now)
                return
        if name == "hand":
            i = fly.nearest(pos, 40)
            if i is not None and fly.frozen_at is None:
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
                self.sound.play("flick")
        elif name == "swatter":
            if not self.swats or now - self.swats[-1][1] > 0.35:
                self.swats.append([pos, now, False])
        elif name == "bomb" and len(self.bombs) < 3:
            self.bombs.append({"p": np.array(pos, float), "v": np.zeros(2), "t": now})
        elif name in ("torch", "cleaner", "freeze"):
            self.torching = True
        elif name == "zapper" and now >= self.zap_ready:
            self._zap(pos, now)
        elif name == "spider" and self.spider is None and not fly.dead:
            self.spider = {"p": np.array([pos[0], CEIL + 4.0]), "state": "hunt", "bite_at": 0.0, "bites": 0, "t": now}
            self.sound.play("drop")
            self.popup((pos[0], CEIL + 70), "A SPIDER!", (200, 200, 210))
        elif name == "sugar" and len(self.sugars) < 3:
            self.sugars.append({"p": np.array(pos, float), "v": 0.0, "left": 1.0})
            self.sound.play("pop")

    def _spray(self, mouse, now: float, kind: str) -> None:
        """Mist toward the fly. Brake cleaner soaks it (dissolves; smell and taste neurons); freeze spray chills it
        (cold-sensing neurons)."""
        fly = self.fly
        nozzle = np.array(mouse, float)
        to_fly = fly.p[THX] - nozzle
        dist = float(np.hypot(*to_fly))
        aim = to_fly / dist if dist > 1 else np.array([1.0, 0.0])
        self.torch_aim = aim
        for _ in range(5):
            a = math.atan2(aim[1], aim[0]) + random.uniform(-0.3, 0.3)
            sp = random.uniform(6, 10)
            self.mist.append([nozzle[0], nozzle[1], math.cos(a) * sp, math.sin(a) * sp, now, random.uniform(0.35, 0.6), kind])
        rel = fly.p - nozzle
        d = np.hypot(*rel.T)
        cosang = (rel @ aim) / np.maximum(d, 1e-6)
        if fly.dissolved_at is not None or fly.frozen_at is not None or not np.any((d < 230) & (cosang > 0.8)):
            return
        fly.last_hit_x = nozzle[0]
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
            for region, side in TORCH_KEYS[:-1]:       # ice on the cuticle
                self.brain.poke(region, side, 0.35)
            self.damage(0.1, "freezing")
            words, col = ("SO COLD!", "BRRRR!", "ICING!"), (190, 230, 255)
        if int(now * 2) != int((now - 1 / 60) * 2):
            self.hits += 1
        if random.random() < 0.02:
            self.popup(fly.p[HEAD] + (0, -60), random.choice(words), col)

    def _zap(self, pos, now: float) -> None:
        """Electric shock: every touch neuron plus current into a random 30% of all neurons for 40 ms (the sim can't
        place a current path, so the shocked neurons are random)."""
        fly = self.fly
        self.zap_ready = now + 0.3
        self.sound.play("zap")
        target = fly.p[THX]
        if np.hypot(*(target - pos)) > 280 or fly.dissolved_at is not None or fly.shattered_at is not None:
            self.bolts.append([np.array(pos, float), np.array(pos, float) + (0, 60), now])
            return
        self.bolts.append([np.array(pos, float), target.copy(), now])
        fly.zap_until = now + 0.3
        fly.char = min(1.0, fly.char + 0.06)
        fly.last_hit_x = pos[0]
        for i in range(N_P):
            fly.impulse(i, np.random.normal(0, 3, 2))
        self.shake_until = now + 0.12
        if fly.dead:
            return
        for region, side in TORCH_KEYS[:-1]:
            self.brain.poke(region, side, 1.0)
        self.brain.poke("all", None, 0.0)
        fly.stun(now, 1.0)
        self.damage(16, "the zapper")
        self.popup(target + (0, -70), random.choice(("BZZZT!", "ZAP!", "KRZZT!")), (200, 235, 255))

    def _shatter(self, now: float) -> None:
        fly = self.fly
        fly.shattered_at = now
        self.shake_until = now + 0.25
        self.sound.play("shatter")
        for i in range(N_P):
            for _ in range(3):
                self.shards.append([fly.p[i, 0], fly.p[i, 1], random.uniform(-9, 9), random.uniform(-14, 2),
                                    random.uniform(0, 6.28), random.uniform(5, 13), random.random() < 0.35])
        self.popup(fly.p[THX] + (0, -60), "SHATTERED!", (200, 235, 255), force=True)

    def _effects(self, now: float) -> None:
        """Brake cleaner, freezing, venom and sugar over time. Their brain damping is a game rule (see docstring)."""
        fly = self.fly
        if self.immortal:                                    # melting and venom stop short and wear off
            fly.melt = min(fly.melt, 0.85)
            if fly.soak < 0.05:
                fly.melt = max(0.0, fly.melt - 0.002)
            fly.venom = max(0.0, fly.venom - 0.002)
        if fly.dissolved_at is None and fly.soak > 0.02:
            fly.melt = min(0.85 if self.immortal else 1.0, fly.melt + 0.0045 * fly.soak)
            fly.soak *= 0.985 if self.immortal else 0.997     # immortal: the solvent evaporates fast
            if not fly.dead:
                self.damage(0.25 * fly.soak, "brake cleaner")
                for region, side in TORCH_KEYS[:-1]:  # the solvent eating the cuticle hits the touch neurons too
                    self.brain.poke(region, side, 0.5 * fly.soak)
        if fly.melt >= 1.0 and fly.dissolved_at is None:
            fly.dissolved_at = now
            self.puff((fly.p[THX, 0], FLOOR - 10), 14, 3)
            self.popup(fly.p[THX] + (0, -60), "DISSOLVED", (170, 230, 255), force=True)
            self.sound.play("squish")
            if not fly.dead:
                self.damage(MAX_HEALTH, "brake cleaner")
        if fly.frozen_at is None and fly.frost > 0:
            if not (self.torching and TOOLS[self.tool][0] == "freeze"):
                fly.frost = max(0.0, fly.frost - 0.0015)   # thawing
            if fly.frost >= 1.0:
                fly.frozen_at = now
                self.popup(fly.p[THX] + (0, -70), "FROZEN SOLID", (190, 230, 255), force=True)
                if not fly.dead:
                    self.damage(MAX_HEALTH, "freezing")
        if not fly.dead:
            self.brain.sedation = max(0.25 * fly.melt ** 2.5,       # solvent, mild at first so it can still flee
                                      0.2 * fly.frost ** 2,         # cold slows neurons
                                      0.3 * fly.venom ** 1.5)       # spider venom
        self._spider(now)
        self._sugar(now)

    def _spider(self, now: float) -> None:
        sp, fly = self.spider, self.fly
        if sp is None:
            return
        if sp["state"] == "hunt":
            if fly.dead or fly.dissolved_at is not None or fly.shattered_at is not None:
                sp["state"] = "leave"
                return
            d = fly.p[THX] + (0, -18) - sp["p"]
            dist = float(np.hypot(*d))
            if dist > 26:
                sp["p"] += d / dist * min(3.4, dist)
            elif now - sp["bite_at"] > 0.7:
                sp["bite_at"] = now
                sp["bites"] += 1
                fly.venom = min(1.0, fly.venom + 0.3)
                fly.last_hit_x = sp["p"][0]
                fly.hurt = 1.0
                self.brain.poke("body", None, 0.9)
                self.brain.poke("legs", "L", 0.6)
                self.brain.poke("legs", "R", 0.6)
                self.damage(16, "a spider")
                self.sound.play("chomp")
                self.popup(sp["p"] + (0, -40), random.choice(("CHOMP!", "BITE!", "SLURP!")), (230, 120, 120))
                if sp["bites"] >= 2 and not fly.wrapped:
                    fly.wrapped = True
                    self.note("WRAPPED  in spider silk")
                if self.immortal and sp["bites"] >= 5:          # it can't be eaten: it breaks out and the spider leaves
                    fly.wrapped, fly.grabbed = False, None
                    sp["state"] = "leave"
                    self.popup(fly.p[HEAD] + (0, -60), "BROKE FREE!", (255, 225, 120), force=True)
                    self.note("BROKE FREE of the silk")
                    return
            if fly.wrapped:
                fly.grabbed = THX
        elif sp["state"] in ("leave", "carry"):
            if fly.wrapped and fly.dead and fly.frozen_at is None:
                sp["state"] = "carry"
            sp["p"][1] -= 2.2
            if sp["state"] == "carry":
                fly.grabbed = THX
            if sp["p"][1] < CEIL - 60:
                self.spider = None
                if fly.wrapped:
                    fly.grabbed = None

    def _sugar(self, now: float) -> None:
        """Sugar: the fly walks over and eats (game rule). Eating drives taste neurons and the PAM reward neurons and
        heals it."""
        fly = self.fly
        for s in self.sugars:
            if s["p"][1] < FLOOR - 8:
                s["v"] += 0.9
                s["p"][1] = min(FLOOR - 8, s["p"][1] + s["v"])
        if not self.sugars or fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None:
            return
        s = min(self.sugars, key=lambda s: abs(s["p"][0] - fly.p[THX, 0]))
        dx = s["p"][0] - fly.p[HEAD, 0]
        on_floor = fly.p[THX, 1] > FLOOR - STAND - 30 and now >= fly.escape_until
        if abs(dx) > 30:
            if on_floor and now >= fly.stun_until:
                fly.facing = 1 if dx > 0 else -1
                fly.walk_until, fly.run = now + 0.2, False
            return
        if not on_floor or s["p"][1] < FLOOR - 10:
            return
        if now >= fly.eating_until:
            self.popup(fly.p[HEAD] + (0, -60), random.choice(("YUM!", "SWEET!", "NOM NOM")), (255, 160, 190))
            self.sound.play("yum")
            self.note("EATING   sugar: taste + PAM reward")
        fly.eating_until = now + 0.4
        s["left"] -= 1 / 240
        self.brain.poke("taste", None, 0.5)
        self.brain.poke("reward", None, 0.4)
        fly.health = min(MAX_HEALTH, fly.health + 0.15)
        if s["left"] <= 0:
            self.sugars.remove(s)

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
        if int(now * 2) != int((now - 1 / 60) * 2):
            self.hits += 1
        if random.random() < 0.02:
            self.popup(fly.p[HEAD] + (0, -60), random.choice(("SIZZLE!", "TSSSS!", "HOT HOT!")), (255, 150, 60))

    def _swat_impact(self, pos, now: float) -> None:
        fly = self.fly
        d = np.hypot(*(fly.p - pos).T) - RADIUS
        near = np.flatnonzero(d < 95)
        self.shake_until = now + 0.18
        self.sound.play("whack")
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
        self.sound.play("boom")
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
        if fly.frozen_at is not None and fly.shattered_at is None and worst:
            self._shatter(now)
        elif worst:
            fly.last_hit_x = pos[0]
            fly.stun(now, 2.4)
            self.damage(32 * worst / 48, "a bomb")
        self.popup((pos[0], pos[1] - 70), "KABOOM!", (255, 160, 60), force=True)

    # --- per frame ---
    def update(self, now: float, mouse) -> None:
        fly, br = self.fly, self.brain
        self.frame += 1
        self.mouse = mouse
        self._environment(now, mouse)
        pin = np.array(mouse, float)
        if fly.wrapped and self.spider is not None:          # carried in the silk under the spider
            pin = self.spider["p"] + (0, 26)
        for i, sp in fly.step(now, pin):
            s = float(np.clip((sp - 9) / 35, 0.05, 1))
            self.hit(i, s)
            if sp > 18:
                self.damage(min(8.0, (sp - 18) * 0.35), "the wall" if fly.p[i, 1] < FLOOR - 30 else "the floor")
            if sp > 14 and fly.p[i, 1] > FLOOR - 30:
                self.puff((fly.p[i, 0], FLOOR - 2), 3)
            if sp > 24 and not fly.dead:
                self.sound.play("bonk", 0.4 + 0.6 * s)
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
        if self.torching and TOOLS[self.tool][0] in ("cleaner", "freeze") and self.report is None:
            self._spray(mouse, now, TOOLS[self.tool][0])
        self._effects(now)
        for sh in self.shards:
            sh[0] += sh[2]
            sh[1] = min(FLOOR - 3, sh[1] + sh[3])
            sh[3] += 0.8
            sh[2] *= 0.98 if sh[1] < FLOOR - 3 else 0.8
            sh[4] += sh[2] * 0.05
        self.bolts = [b for b in self.bolts if now - b[2] < 0.25]
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
        parts[3] = max(parts[3], self.pain_parts[3] * 0.96)
        self.pain_parts += (parts - self.pain_parts) * 0.15
        self.pain += (float(br.pain_index(parts)) - self.pain) * 0.15
        self.reward += (float(np.clip((br.level("reward") - 1.0) / 0.6, 0, 1)) * 100 - self.reward) * 0.1
        if not fly.dead and self.pain > 25 and self.frame % 3 == 0:
            br.poke("punish", None, self.pain / 100)        # pain drives the punishment dopamine neurons (game rule)
        self._vision(now, mouse)
        self._scents(now, mouse)
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
                floor = 1.0 if self.immortal else 0.0          # immortal: it can be hurt, never killed
                fly.health = max(floor, fly.health - self.pending_damage)
                self.last_damage = now
                if fly.health <= 0:
                    self._die(now)
            elif self.immortal and now - self.last_damage > 1.5:
                fly.health = min(MAX_HEALTH, fly.health + 0.1)   # heals ~6 health/s once you stop
        self.pending_hits.clear()
        self.pending_damage = 0.0
        if fly.dead:
            if self.report is None and now - fly.dead_at > AUTOPSY_DELAY:
                self.report = self._autopsy(now)
                self.death_frames = list(self.frames)
            return

        # reactions read from the descending neurons
        can_fly = fly.grabbed is None and not fly.wrapped and fly.frost < 0.5 and fly.melt < 0.3 and fly.venom < 0.5
        free = fly.grabbed is None and now >= fly.escape_until
        can_fly = can_fly and fly.wet <= 0 and len(fly.stuck) < 2
        lv = {n: br.level(n) for n in ("jump", "run", "kick", "walk", "back", "turn_l", "turn_r", "fly", "escape")}
        fly.power = lv["fly"]
        away = 1 if fly.p[THX, 0] >= fly.last_hit_x else -1
        if lv["escape"] > THRESH["escape"] and now >= fly.escape_ready and fly.grabbed is None and can_fly:
            fly.stun_until = 0.0                             # giant fiber escape: it saw something coming
            fly.last_hit_x = self.threat_x
            fly.escape(now)
            self.note(f"DODGE    giant fiber DNp01 x{lv['escape']:.1f}")
            self.popup(fly.p[HEAD] + (0, -60), "DODGE!", (170, 255, 200))
            self.sound.play("dodge")
        elif lv["jump"] > THRESH["jump"] and now >= fly.escape_ready and free and can_fly:
            fly.stun_until = 0.0                             # the reflex beats the dizziness
            fly.escape(now)
            self.note(f"FLY AWAY head-touch DNs x{lv['jump']:.1f}")
            self.popup(fly.p[HEAD] + (0, -60), "YIKES!", (160, 230, 255))
        elif (lv["fly"] > THRESH["fly"] and now >= fly.escape_ready and free and can_fly and now >= fly.stun_until
              and now >= fly.eating_until):
            fly.escape(now, seconds=random.uniform(2.5, 4.0), wander=True)
            self.note(f"TAKE OFF DNg02 x{lv['fly']:.2f}")
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
        self._memory_behavior(now, free, can_fly)
        lamp_idle = free and now >= fly.escape_until and now >= fly.stun_until and now >= fly.walk_until
        if ARENAS[self.arena_i] == "lamp" and lamp_idle and now >= self.photo_ready:
            self.photo_ready = now + random.uniform(3.0, 6.0)   # drawn to the light (game rule)
            fly.facing = 1 if LAMP[0] > fly.p[THX, 0] else -1
            if can_fly and random.random() < 0.6:
                fly.escape(now, seconds=random.uniform(3.0, 5.0), wander=True)
                self.note("TO LIGHT flies to the lamp")
            else:
                fly.walk_until, fly.run = now + 1.5, False
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
        self.sound.play("death")
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
        names = ["whole brain"] + [n for n, _ in POPS] + ["head", "body", "legs", "wing", "heat", "cold", "smell", "taste", "reward", "jump", "run", "fly"]
        pretty = {"head": "touch: head", "body": "touch: body", "legs": "touch: legs", "wing": "touch: wings", "heat": "heat sensors",
                  "cold": "cold sensors", "smell": "smell ORNs", "taste": "taste neurons", "reward": "PAM reward DANs",
                  "jump": "DNs: jump group", "run": "DNs: run group", "fly": "DNs: DNg02 flight"}
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
        self._draw_arena_back(arena, now)

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
            col = (200, 230, 250) if m[6] == "cleaner" else (225, 245, 255)
            aacircle(arena, (m[0], m[1]), 5 + 22 * e, (*col, int((90 if m[6] == "cleaner" else 130) * (1 - e))))
        if self.torching and TOOLS[self.tool][0] in ("cleaner", "freeze") and self.report is None and mouse[0] < PLAY_W:
            aim = getattr(self, "torch_aim", np.array([1.0, 0.0]))
            m = np.array(mouse, float)
            body = (200, 40, 40) if TOOLS[self.tool][0] == "cleaner" else (60, 130, 210)
            thick_line(arena, m - aim * 78, m - aim * 12, 26, body)                     # the can
            thick_line(arena, m - aim * 60, m - aim * 40, 27, (235, 235, 240))         # label band
            thick_line(arena, m - aim * 12, m - aim * 2, 8, (60, 60, 66))              # nozzle
        for sh in self.shards:                           # ice and fly fragments
            c, s_ = math.cos(sh[4]), math.sin(sh[4])
            r = sh[5]
            pts = [(sh[0] + r * c, sh[1] + r * s_), (sh[0] - r * 0.6 * s_, sh[1] + r * 0.6 * c), (sh[0] - r * c * 0.8, sh[1] - r * s_ * 0.5)]
            aapoly(arena, pts, (150, 100, 50) if sh[6] else (200, 235, 255, 200))
        for b in self.bolts:                             # lightning
            a, z = b[0], b[1]
            pts = [a]
            for k in range(1, 7):
                q = a + (z - a) * k / 7
                pts.append(q + np.random.normal(0, 12, 2))
            pts.append(z)
            for w_, col in ((7, (120, 170, 255, 120)), (3, (235, 245, 255))):
                pygame.draw.lines(arena, col[:3], False, [tuple(p) for p in pts], w_)
        if self.spider is not None:
            self._draw_spider(arena, now)
        for s in self.sugars:
            k_ = max(0.35, s["left"])
            x, y = s["p"]
            e = 11 * k_
            aapoly(arena, [(x - e, y - e * 0.4), (x, y - e), (x + e, y - e * 0.4), (x, y + e * 0.2)], (250, 250, 255))
            aapoly(arena, [(x - e, y - e * 0.4), (x, y + e * 0.2), (x, y + e), (x - e, y + e * 0.5)], (215, 215, 228))
            aapoly(arena, [(x, y + e * 0.2), (x + e, y - e * 0.4), (x + e, y + e * 0.5), (x, y + e)], (185, 185, 205))
        self._draw_arena_front(arena, now)
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
        if not self._overlay_open() and TOOLS[self.tool][0] != "hand" and mouse[0] < PLAY_W and mouse[1] < FLOOR:
            gfxdraw.aacircle(arena, mouse[0], mouse[1], 10, (255, 255, 255))
            pygame.draw.line(arena, (255, 255, 255), (mouse[0] - 14, mouse[1]), (mouse[0] + 14, mouse[1]))
            pygame.draw.line(arena, (255, 255, 255), (mouse[0], mouse[1] - 14), (mouse[0], mouse[1] + 14))
        if self.report is not None:
            self._draw_autopsy(arena, now)
        elif self.big_view:
            self._draw_big_view(arena)
        if self.surgery_open:
            self._draw_surgery(arena)
        if self.help_open:
            self._draw_help(arena)

        shake = (0, 0)
        if now < self.shake_until:
            shake = (random.randint(-7, 7), random.randint(-5, 5))
        scr.blit(arena, shake)
        self._draw_brain(now)

    def _draw_spider(self, surf, now: float) -> None:
        sp = self.spider
        x, y = sp["p"]
        pygame.draw.aaline(surf, (220, 220, 225), (x, CEIL), (x, y - 8))
        for sgn in (-1, 1):
            for k in range(4):
                a = (k - 1.5) * 0.5
                wig = 3 * math.sin(now * 14 + k * 1.7 + sgn)
                knee = (x + sgn * 20 * math.cos(a), y - 12 + 10 * math.sin(a) + wig)
                foot = (x + sgn * 30 * math.cos(a), y + 6 + 12 * math.sin(a) - wig)
                thick_line(surf, (x, y), knee, 2.5, (30, 28, 32))
                thick_line(surf, knee, foot, 2, (30, 28, 32))
        aacircle(surf, (x, y + 4), 13, (38, 34, 40))
        aapoly(surf, [(x - 3, y), (x + 3, y), (x, y + 6)], (200, 40, 40))
        aapoly(surf, [(x - 3, y + 12), (x + 3, y + 12), (x, y + 6)], (200, 40, 40))
        aacircle(surf, (x, y - 10), 8, (30, 28, 32))
        for dx in (-3, 3):
            aacircle(surf, (x + dx, y - 12), 1.6, (230, 60, 60))

    def _fly_state(self, now: float) -> str:
        f = self.fly
        if f.shattered_at is not None:
            return "shattered"
        if f.dissolved_at is not None:
            return "dissolved"
        if f.frozen_at is not None:
            return "frozen solid"
        if f.dead:
            return "dead"
        if f.stuck:
            return "stuck on flypaper"
        if f.arena == "pool" and (f.p[:, 1] > WATER_Y).any() and now >= f.escape_until:
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
        self._text(surf, self._fly_state(now).upper(), (22, 12), AMBER, self.f_head)
        self._text(surf, f"hits {self.hits}   kills {self.kills}", (22, 42), TEXT, self.f_text)
        # health
        bw, bx, by = 300, PLAY_W // 2 - 150, 16
        frac = fly.health / MAX_HEALTH
        col = (230, 185, 60) if self.immortal else S_GOOD if frac > 0.5 else S_WARN if frac > 0.25 else S_CRIT
        pygame.draw.rect(surf, (10, 12, 18), (bx - 4, by - 4, bw + 8, 26), border_radius=13)
        if frac > 0:
            pygame.draw.rect(surf, col, (bx, by, max(18, int(bw * frac)), 18), border_radius=9)
            pygame.draw.rect(surf, tuple(min(255, c + 60) for c in col), (bx + 6, by + 3, max(6, int(bw * frac) - 12), 4), border_radius=2)
        label = "DEAD" if fly.dead else f"HEALTH {fly.health:.0f}" + ("  IMMORTAL" if self.immortal else "")
        self._text(surf, label, (PLAY_W // 2, by + 9), INK, self.f_bold, "center")
        bits = [f"arena: {ARENAS[self.arena_i]} (E)"]
        if self.brain.surgery:
            bits.append("SURGERY ON (O)")
        if self.sound.muted:
            bits.append("muted (M)")
        bits.append(getattr(self, "hint_extra", "F11 fullscreen   H help"))
        hint = self.f_small.render("   ".join(bits), True, AMBER if self.brain.surgery else TEXT)
        box = hint.get_rect(topright=(PLAY_W - 14, 44)).inflate(14, 6)
        pygame.draw.rect(surf, (10, 12, 18, 170), box, border_radius=6)
        surf.blit(hint, hint.get_rect(center=box.center))
        if self.saved_msg and now - self.saved_msg[1] < 4:
            self._text(surf, self.saved_msg[0], (PLAY_W // 2, 66), (170, 230, 190), self.f_small, "midtop")
        self._draw_pain(surf)
        self._draw_reward(surf)
        self._draw_memory(surf)

    def _draw_reward(self, surf) -> None:
        x, y, w, h = 10, 308, 236, 64
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 180), card.get_rect(), border_radius=10)
        surf.blit(card, (x, y))
        r = self.reward
        word, col = ("bliss", (255, 120, 170)) if r >= 70 else ("happy", (255, 160, 190)) if r >= 30 else \
            ("pleased", (220, 190, 200)) if r >= 8 else ("neutral", LABEL)
        self._text(surf, "REWARD", (x + 12, y + 8), LABEL, self.f_small)
        self._text(surf, f"{r:.0f}", (x + 72, y + 3), INK, self.f_head)
        self._text(surf, word, (x + w - 12, y + 6), col, self.f_bold, "topright")
        pygame.draw.rect(surf, (30, 36, 48), (x + 12, y + 32, w - 24, 10), border_radius=5)
        if r > 0.5:
            pygame.draw.rect(surf, col, (x + 12, y + 32, max(8, int((w - 24) * r / 100)), 10), border_radius=5)
        lvl = self.brain.level("reward")
        self._text(surf, f"PAM dopamine neurons x{lvl:.2f} calm", (x + 12, y + 46), LABEL, self.f_small)

    def _draw_pain(self, surf) -> None:
        x, y, w, h = 10, 78, 236, 222
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
        for label, v in zip(("touch overload", "heat / cold", "chemical", "DN alarm", "body relay"), self.pain_parts):
            self._text(surf, label, (bx, yy - 3), TEXT, self.f_small)
            self._bar(surf, bx + 112, yy, bw - 146, float(v), (170, 176, 188))
            self._text(surf, f"{v * 100:3.0f}", (bx + bw, yy - 3), TEXT, self.f_small, "topright")
            yy += 15
        tr = self.pain_trace
        sy, sh = y + 162, 22
        pygame.draw.rect(surf, (22, 26, 34), (bx, sy, bw, sh), border_radius=3)
        if len(tr) > 2:
            pts = [(bx + k * bw / 719, sy + sh - tr[k] / 100 * (sh - 2)) for k in range(0, len(tr), 3)]
            if len(pts) > 1:
                pygame.draw.aalines(surf, S_CRIT, False, pts)
        self._text(surf, f"peak {self.pain_peak:.0f}   maxed {self.pain_max_s:.1f}s", (bx, y + h - 34), LABEL, self.f_small)
        lvl = PAIN_LEVELS[self.pain_level][0]
        self._text(surf, f"P  {lvl} ({self.brain.pain_neurons():,} neurons)", (bx, y + h - 18),
                   AMBER if self.pain_level else TEXT, self.f_small)

    def _draw_toolbar(self, surf) -> None:
        bw, gap = 80, 6
        x0 = (PLAY_W - bw * len(TOOLS) - gap * (len(TOOLS) - 1)) // 2
        self.tool_rects = []
        name, label, hint = TOOLS[self.tool]
        key = "0" if self.tool == 9 else str(self.tool + 1)
        self._text(surf, f"{key}  {label}: {hint}", (PLAY_W // 2, FLOOR + 10), INK, self.f_bold, "midtop")
        for k, (name, label, hint) in enumerate(TOOLS):
            r = pygame.Rect(x0 + k * (bw + gap), FLOOR + 38, bw, 64)
            self.tool_rects.append(r)
            on = k == self.tool
            card = pygame.Surface(r.size, pygame.SRCALPHA)
            pygame.draw.rect(card, (60, 46, 22, 230) if on else (16, 18, 24, 210), card.get_rect(), border_radius=10)
            surf.blit(card, r)
            pygame.draw.rect(surf, AMBER if on else BORDER, r, 2, border_radius=10)
            draw_icon(surf, name, (r.centerx, r.y + 24), AMBER if on else TEXT)
            self._text(surf, "0" if k == 9 else str(k + 1), (r.x + 7, r.y + 4), LABEL, self.f_small)
            self._text(surf, label, (r.centerx, r.y + 44), AMBER if on else INK, self.f_small, "midtop")

    def _draw_brain(self, now: float) -> None:
        scr, br = self.screen, self.brain
        x0 = PLAY_W
        pa = getattr(self, "panel_alpha", 255)
        pygame.draw.rect(scr, (*PANEL_BG, pa), (x0, 0, W - x0, H))
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
            scr.blit(self._panel_image(self._view_surface("panel")), (x, 54))
            self._hud_overlay(scr, self.view_rect, now, small=True)
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
        y = self._card(scr, x, y, bw, "DESCENDING NEURONS", "x calm baseline", len(MOTOR))
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
        for t, msg in reversed(self.log[-(4 if H >= 740 else 3):]):
            age = now - t
            col = AMBER if age < 1.0 else TEXT if age < 5 else DIM
            self._text(scr, f"{age:4.1f}s  {msg}", (x, y), col, self.f_small)
            y += 15
        self._text(scr, f"sim {br.steps_per_s:4.0f} steps/s  {br.sim.last_step_ms:4.1f} ms/step",
                   (x, H - 18), DIM, self.f_small)

    def _card(self, scr, x, y, w, title, unit, rows) -> int:
        h = 24 + rows * 16
        pygame.draw.rect(scr, (*CARD, min(255, getattr(self, "panel_alpha", 255) + 40)), (x - 4, y - 4, w + 8, h + 4), border_radius=8)
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
        self.save_rects = []
        for k, (label, what) in enumerate((("SAVE IMAGE", "png"), ("SAVE DEATH GIF", "gif"))):
            r = pygame.Rect(card.right - 196 + k * 88 - (0 if k == 0 else 4), card.y + 70, 84 if k == 0 else 110, 26)
            pygame.draw.rect(surf, (44, 50, 64), r, border_radius=7)
            self._text(surf, label, r.center, INK, self.f_small, "center")
            self.save_rects.append((r, what))

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
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            if self.help_open or self.surgery_open or self.big_view:   # Esc closes an overlay before it quits
                self.help_open = self.surgery_open = self.big_view = False
                return True
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_h:
                self.help_open = not self.help_open
            elif ev.key == pygame.K_o:
                self.surgery_open = not self.surgery_open
            elif ev.key == pygame.K_e:
                self.arena_i = (self.arena_i + 1) % len(ARENAS)
                self.fly.stuck.clear()
                self.note(f"ARENA    {ARENAS[self.arena_i]}")
            elif ev.key == pygame.K_m:
                self.sound.muted = not self.sound.muted
                for slot in list(self.sound.loops):
                    self.sound.loop(slot, None)
            elif ev.key == pygame.K_s:
                self.save_png()
            elif ev.key == pygame.K_g:
                self.save_gif()
            elif ev.key in TOOL_KEYS:
                self.tool = TOOL_KEYS.index(ev.key)
            elif ev.key == pygame.K_r:
                self.new_fly()
            elif ev.key == pygame.K_b:
                self.big_view = not self.big_view
            elif ev.key == pygame.K_i:
                self.immortal = not self.immortal
                self.note(f"IMMORTAL {'on: it can feel pain but never die' if self.immortal else 'off'}")
                self.popup(self._above_head(), "IMMORTAL!" if self.immortal else "MORTAL", (255, 225, 120), force=True)
            elif ev.key == pygame.K_p:
                self.pain_level = (self.pain_level + 1) % len(PAIN_LEVELS)
                self.brain.set_pain_level(self.pain_level)
                self.note(f"PAIN     {PAIN_LEVELS[self.pain_level][0]}: {self.brain.pain_neurons():,} neurons")
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.help_open:
                self.help_open = False
                return True
            if self.surgery_open:
                for r, k, mode in self.surgery_buttons:
                    if r.collidepoint(ev.pos):
                        if k < 0:
                            self.surgery_modes = [0] * len(SURGERY)
                            self.type_ops = {}
                            self._apply_surgery()
                            self.note("SURGERY  cleared")
                        else:
                            self._set_surgery(k, mode)
                return True
            if self.report is not None:
                if getattr(self, "new_fly_rect", None) and self.new_fly_rect.collidepoint(ev.pos):
                    self.new_fly()
                for r, what in getattr(self, "save_rects", []):
                    if r.collidepoint(ev.pos):
                        self.save_png() if what == "png" else self.save_gif(self.death_frames)
                return True
            if self.view_rect.collidepoint(ev.pos):
                self.big_view = not self.big_view
                return True
            if self.big_view:
                if self.inspect is not None:
                    for r, mode in self.inspect_buttons:
                        if r.collidepoint(ev.pos):
                            self.type_ops[self.inspect["type"]] = mode
                            self._apply_surgery()
                            self.note(f"SURGERY  {self.inspect['type']}: {'off' if mode < 0 else 'on' if mode > 0 else 'normal'}")
                            return True
                if self.big_rect.collidepoint(ev.pos):
                    self._inspect_at(ev.pos)
                return True
            for k, r in enumerate(getattr(self, "tool_rects", [])):
                if r.collidepoint(ev.pos):
                    self.tool = k
                    return True
            if ev.pos[0] < PLAY_W:
                self.use_tool(ev.pos, now)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if not self.fly.wrapped:
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
        brain = Brain(g, sim)
        out["stage"] = "placing neurons"
        pain_groups = [brain.col[n] for n in (*TOUCH, "heat", "cold", "smell", "taste", "body_extra")]
        pain_mask = np.isin(brain.det_id, pain_groups) | (brain.pop_id == [n for n, _ in POPS].index("ascending"))
        out["view"] = BrainView(soma, weights, pain_mask)
        out["stage"] = "waking the fly up"
        brain.warmup()
        out["brain"] = brain
    except Exception as e:  # shown on the loading screen
        out["error"] = f"{type(e).__name__}: {e}"


def main() -> int:
    smoke = float(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0.0  # build check: run N s, exit
    pygame.mixer.pre_init(Sound.RATE, -16, 1, 512)
    if "--2d" not in sys.argv:                           # first person 3D by default; 2D if OpenGL 3.3 isn't there
        try:
            import kick3d
        except Exception as e:                           # e.g. moderngl missing in a source checkout
            print(f"3D unavailable ({e}); starting the 2D game")
        else:
            pygame.init()
            shot = sys.argv[sys.argv.index("--smoke") + 2] if smoke and len(sys.argv) > sys.argv.index("--smoke") + 2 else None
            try:
                return kick3d.run(smoke, shot, "--fullscreen" in sys.argv)
            except (kick3d.moderngl.Error, pygame.error) as e:
                print(f"3D failed ({e}); starting the 2D game")
                pygame.display.quit()
    os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "linear")   # smooth when scaled, not blocky
    pygame.init()
    pygame.display.set_caption("Kick the Fly")
    # SCALED: the game always draws at 1280x760 and SDL scales that to the window or the whole screen, keeping the
    # aspect ratio (black bars if needed) and mapping the mouse back, so it can go fullscreen at any resolution.
    screen = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
    desk = pygame.display.get_desktop_sizes()[0] if pygame.display.get_desktop_sizes() else (W, H)
    if "--fullscreen" in sys.argv or desk[0] < W or desk[1] < H + 60:   # asked for, or the window wouldn't fit
        pygame.display.toggle_fullscreen()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui,consolas", 22)
    state: dict = {"stage": "starting"}
    threading.Thread(target=load_brain, args=(state,), daemon=True).start()
    t0 = time.perf_counter()
    while "brain" not in state:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and (ev.key == pygame.K_F11 or
                                              (ev.key == pygame.K_RETURN and ev.mod & pygame.KMOD_ALT)):
                pygame.display.toggle_fullscreen()
                continue
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
            try:
                from PIL import Image  # noqa: F401  (GIF saving works in this build)
                gif = "gif ok"
            except ImportError:
                gif = "no gif"
            status = f"smoke ok: {brain.n:,} neurons, {brain.steps_per_s:.0f} steps/s, sound {game.sound.ok}, {gif}"
            print(status)
            if len(sys.argv) > sys.argv.index("--smoke") + 2:   # optional screenshot path; the exe has no console
                shot = sys.argv[sys.argv.index("--smoke") + 2]
                pygame.image.save(screen, shot)
                with open(shot + ".txt", "w") as f:
                    f.write(status)
            break
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and (ev.key == pygame.K_F11 or
                                              (ev.key == pygame.K_RETURN and ev.mod & pygame.KMOD_ALT)):
                pygame.display.toggle_fullscreen()
                continue
            running = game.handle(ev, now) and running
        game.update(now, (min(mouse[0], PLAY_W - 5), mouse[1]))
        game.draw(now, mouse)
        if game.frame % 4 == 0:                      # rolling footage for G / the death GIF
            game.capture()
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
