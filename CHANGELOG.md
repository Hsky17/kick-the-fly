# Changelog

## [Unreleased]

### Added
- **Pluggable Simulation Backends:** Pluggable compute backend abstraction (`auto`, `cpu`, `numba`, `torch-cuda`, `torch-rocm`) with graceful fallback to standard NumPy CPU. Includes JIT-compiled Numba kernels (`@njit(fastmath=True)`) and PyTorch CUDA/ROCm/CPU tensors while guaranteeing bit-exact determinism for same-seed validation runs.
- **Dynamic Fly Cap & Multi-Fly Swarm Scaling:** Raised the default fly cap from 16 to dynamically adapt to the active compute backend (16 on baseline CPU, 24 on PyTorch CPU, 32 on Numba, and 64 on GPU). Added **F** cycling key with a 5-second manual focus override before resuming nearest-fly tracking.
- **Arbitrary-Duration Video Recording:** Fullscreen MP4/WebM video streaming directly to `ffmpeg` via stdin pipe with automatic fallback to Pillow animated GIF when ffmpeg is unavailable. Accessible via hotkey **Shift+R** (or capital **R**) with an on-screen HUD badge `[REC mm:ss]`, saving to `videos/` or a custom path via `--record-video [PATH]`.
- **Real Neuron Morphology (SWC Skeletons):** Live fetching and caching of authentic EM reconstruction skeletons from Janelia neuPrint (MaleCNS v1.0) for landmark neuron classes (MBONs, Kenyon cells, DNa02 steering, Giant Fiber DNp01) with local caching in `data/skeletons/` and synthetic fiber fallback.
- **Backend & Device Performance Tracking:** Extended `--benchmark`, Lab dashboard, crash reports, save states, and NWB export metadata to record active simulation backend, device name, throughput (neurons/s and syn-evals/s), and execution precision.

### Performance
- **Sparse Matmul & Precision Optimization:** Configurable simulation precision (`float32` default vs `float64`) via `brain.dtype` setting and `--dtype` CLI flag. Verified bit-exact determinism across CSR/CSC layouts with 2x memory reduction on modern compute hardware.
- **Simulation Loop Profiler:** Added dedicated sim loop profiler reporting breakdown across sparse matmul, state updates, plasticity, readout, and rendering across 1, 8, and 16 flies.

## 2.7.2 (2026-09-18)

### Fixed
- **The blowtorch flame and the brake cleaner and freeze sprays now come out of the tool in your hand** (3D). They
  were started 0.7 m behind your head and flew forward through the camera. Thrown bombs, sugar and alcohol, the
  zapper and the laser also start at your hand now. The fire point follows the tool's nozzle on screen at any field
  of view.
- **The laser's key shows as "=" on the toolbar** instead of "12". It always worked on =; `=` is now also reserved so
  it can't be bound to another action.

## 2.7.1 (2026-09-18)

### Fixed
- **The Linux AppImage started in 2D on current distros** (2.6.0 and 2.7.0, seen on Arch with Mesa radeonsi): it
  bundled the C++ runtime and X11 client libraries from its Ubuntu 22.04 build machine, which are too old for a new
  graphics driver, so OpenGL failed to load on Wayland and X11 alike. Those libraries now always come from the host,
  as the AppImage project recommends, and the build fails if any of them slip back in. Native Wayland is still used
  first, with X11 as the fallback.

## 2.7.0 (2026-09-17)

### Added
- **Two outdoor arenas (3D), Open field and Orchard,** selectable with E, in Settings > Brain > Arena, and with `--arena`; the room stays the default.
  - *Open field:* 30 m x 30 m of ground, rocks, grass and sky, with much larger bounds than the room and no ceiling for the fly. Steady wind drives the real JO-C/E wind neurons of each antenna, and sunlight drives the photoreceptors; wind direction and speed and the sun's azimuth and elevation are Lab parameters. Escapes carry the fly away outdoors; a fly more than 26 m from you or 15 m up is lost, and **J** calls it back.
  - *Orchard:* 24 fruit trees. The fly flies to a ripe fruit, lands and feeds, which drives the real sugar-pathway taste and PAM reward neurons exactly as the sugar tool does and heals it; fermented fruit act like the alcohol tool. Each fruit holds a limited number of feeds (default 4), shrinks and browns as it's eaten, then drops, and grows back after about 75 s, staggered, with a per-tree cap (default 4). The fruit, trees and flying to them are game rules and are tagged RULE. Several flies compete only through the looming and touch neurons they already had.
  - Feeds per fruit, regrow time and the cap are Lab parameters, valid protocol `params`, and the new headless `assay: orchard` (`protocols/orchard-feeding.yaml`) runs the feeding schedule reproducibly.
  - The arena is saved in `config.toml`, in save states (by name) with the orchard's fruit, in export metadata with its weather and fruit settings, and in the `--smoke` status line.
- **E-PG compass, second test: steady directional wind** (`validation: epg_compass_wind`). The open field's wind, through the arena's own transduction, from 8 directions, with the visual test's pass criteria fixed beforehand and nothing tuned. It fails: contrast 1.81x (1.71x without wind; 3.0x needed), persistence 101 ms (500 ms needed), direction tracking |r| 0.46 vs 0.40 for shuffled directions (p = 0.17). No compass HUD.
- **Repo restructure:** the ~25 root modules moved into a `kickthefly/` package (`core`, `sim`, `game`, `ui`, `lab`, `data`) with `git mv`. `python kick_the_fly.py` still works through a shim; `python -m kickthefly` does the same. `CONTRIBUTING.md` says where code goes.
- **NWB export:** recordings (and protocols with `nwb: true` or `--nwb`) as one Neurodata Without Borders file with units, rates, stimuli, events, kinematics, surgery, arena, KC->MBON weights before and after, and full metadata with the MaleCNS v1.0 / CC BY 4.0 citation. Optional `pynwb`, not bundled.
- **Synapse threshold slider and report** (Lab > Connectome robustness; `--threshold-sweep`).
- **Sign-flip stress test** on the dataset's transmitter predictions and confidence, with per-neuron flips from the inspector (`--signflip-test`). The brain pack now carries each neuron's transmitter, its confidence and where it came from.
- **Critical path finder** (Lab; `--critical-path TARGET`), resumable, with one-click "apply this lesion".
- **Optogenetics Laser (Lab > Laser):** In-world aimable beam that activates or silences selected cell types directly in real time, replacing menu-only surgery for rapid targeted stimulation/silencing experiments.
- **Psychometrics Curve Generator (Lab > Psychometrics):** Sweeps stimulus parameters across continuous ranges, executes N trials with 95% confidence intervals, and exports publication-ready vector figures (PDF-1.4, SVG) and raw CSV data.
- **E-PG Compass Validation Test (Lab > Validation):** Negative validation assay testing whether a driven E-PG wedge forms a persistent head-direction bump (it doesn't: no self-sustaining bump, so no compass HUD is displayed).
- **Classroom Mode & Lecture Protocols (Lab > Classroom):** Self-contained educational modules with guided steps, hypothesis prompts, and bundled interactive YAML lecture protocols (`protocols/lecture_*.yaml`).
- **Mystery Defect / Reverse Brain Surgery Challenge (Play mode):** Curated circuit silencing challenges without jargon where players test the fly with tools, request hints, and deduce missing neural circuits.
- **Predict-the-Neuron Minigame (Play mode):** Reaction minigame restricted strictly to descending motor readouts (jump, run, kick, back up, take off) where players predict upcoming movements from motor surge cues.
- **Escape-Room Arena (Play mode):** Multi-hazard emergent navigation gauntlet combining fan wind, flypaper strip, and hot lamp overhead; reach the sugar dish to stop the speedrun timer and generate a tamper-evident verification code (`KTF-<SEED>-<TIME>-<SIG>`).
- **Neural Clamp (Lab > Neural clamp):** Record reference spike trains from calibrated runs (looming, antennal grooming, sweet taste, calm baseline) and replay forced spikes into altered connectomes (lesion, threshold pruning, transmitter sign flips) to isolate structural changes from sensory feedback. Side-by-side activity diff table and export. Explicit dynamic clamp notice in UI and exports that forced spikes override membrane state and break closed-loop feedback.
- **Connectome Diff Mode (Lab > Connectome diff):** Run two flies (reference vs perturbed) side-by-side with identical seeds, initial states, and sensory inputs in lockstep. Real-time region-by-region activity divergence with autopsy-style diverging bar charts, timeline tracking the moment trajectories split, and CSV/JSON export.
- **Hemifield & Hemisphere Silencing:** Brain surgery presets for unilateral visual pathways (`LC10`, `LPLC2`, `LC4`, `LPTC`, `VS`, `HS`) and whole hemibrains. Validated behavioral consequences: failure to dodge blind-side looming, asymmetric steering bias (-1.8 Hz vs +0.10 Hz baseline), and broken 1v1 duel aiming on the blind side. Reported strictly as connectome wiring outcomes.
- **Global Inhibition Block / Picrotoxin (Lab > Robustness > Inhibition block):** 0-100% severity slider scaling inhibitory synaptic weights (`inhibition_scale = 1 - severity`). Emergent runaway excitation (>30 Hz brain-wide mean) without scripted seizures; before/after firing rate distributions; convulsion twitching and severity labels tagged as game rules.
- **Research findings in the docs:** the synapse threshold sweep (dropping every connection below 10 synapses removes 73.6% of connections and every input of 10,151 neurons; all 4 validated behaviors survive), sign flips (looming and antennal grooming survive 3/3 trials, sugar -> MN9 fails 3/3), and the looming critical path (LC4 -50%, LPLC2 -43%).

### Changed
- Experimental activation (`simcore.drive`) is its own current, added to surgery instead of replacing it, so a silenced cell type stays silenced when an assay drives it.
- Outdoor-scale rendering: static scenery matrices are built once, instance data is packed in bulk, and collision boxes far from the fly are skipped. With one fly every arena runs at real time and 62 fps; with 8 flies the outdoor arenas run within ~20% of the room (see README > Performance).
- Lab hub scrolls; Model assumptions gains entries for weak connections, transmitter signs, outdoor transduction, the orchard, the alcohol/zapper scent overlap and the compass.

### Fixed
- Dragging across the big brain view no longer freezes the game (the view is re-projected on the brain-view thread, not on every mouse move), and clicking a neuron no longer crashes the inspector.
- Fullscreen scales correctly in 2D and 3D: the 3D window opens at the display's size instead of toggling after creation, F11 works in both, and the 3D HUD is never shorter than 760 units (the bottom toolbar used to be cut off at 1440p).
- The laser works in the 3D game (beam, hit feedback, viewmodel, HUD badge), pulses finish instead of stopping when the mouse is released, and several flies no longer overwrite each other's laser state.
- `--autopilot` and `--mirror-weights` crashed at startup (`Config` doesn't support item assignment).
- Holding the laser near the fly crashed the 3D game (its scent group is empty).
- A packaged build with a pack older than 2.6 crashed on startup instead of leaving regions unassigned.
- README numbers from the previous session corrected against re-measurement (threshold 10 cuts off 10,151 neurons' input, not 12,234; the sign-flip looming effect is x13.2, not x10.9).

## 2.6.0 (2026-09-17)

### Added
- **Pause menu and Settings.** Esc opens Resume, Challenges or Lab tools, Settings, Save State, Load State, Mode and Quit (with confirmation). Settings has Graphics, Audio, Brain, Controls and Accessibility tabs with hover tooltips, sliders that take typed values, per-tab reset, rebindable keys with conflict swapping, and Connectome / Game rule tags on brain settings. Saved to `config.toml`.
- **Play and Lab modes.** Play (default) has three challenges with best scores: Teach it to pick the right door (T-maze), How close can you sneak? (looming escape) and Find its sweet tooth (sugar response). Lab adds the validation dashboard, assays and repeated trials with 95% CIs and same-seed controls, live parameters, recording and export, and protocols.
- **Real vs rule tags** on reactions and popups (on by default in Lab).
- **Validation suite** (`--validate`, `pytest`) with held-out seeds and fixed criteria. Passes: looming -> giant fiber, sugar-pathway taste neurons -> MN9, antennal JO-C/E -> aDN1/aDN2, T-maze conditioning. Fails, reported as such: MDN -> backward walking, aDN -> front-leg motor neurons.
- **Real-science cards** in Play, the first time a validated behavior happens (only for passing tests).
- **Time controls:** pause (Z), slow motion 0.1x-1x ([ ]), single step (.), with an on-screen badge; the room and all brains slow together.
- **Save states:** the whole simulation in a versioned, platform-independent `.ktfsave` file.
- **Protocols:** YAML stimulus or assay protocols, from Lab > Protocols or `--headless --protocol FILE`, with spike/rate CSV and npz exports and metadata.
- **GROOM and PROBOSCIS reactions** read from aDN1/aDN2 and MN9; the proboscis extends while eating sugar.
- **Accessibility:** blue/yellow and high-contrast brain palettes, reduced flashing, larger text.
- **Linux:** XDG Base Directory locations (with a one-time copy of older data), native Wayland with automatic X11 fallback and `--backend`, AppStream metadata and desktop file in the AppImage, an AUR `kickthefly-bin` PKGBUILD.
- **Windows:** per-monitor DPI awareness, Known Folders for Documents and Pictures, version info in the exe's properties.
- **Crash reports** with version, seed, OS, session type, video driver, GPU, OpenGL and driver versions, also written to the per-user state folder.
- **Release workflow** that builds the brain pack, runs all tests on Ubuntu 22.04 and Windows, builds the AppImage and exe, and publishes both with SHA256SUMS.

### Changed
- Esc no longer quits; it closes a panel or opens the pause menu.
- A missing OpenGL 3.3 is logged clearly before falling back to 2D.
- The brain pack also stores subclass labels and FlyEM body IDs (all existing arrays unchanged, so saved training memory still loads).
- The multiple-flies notes in the README match the game (up to 16 flies; the panel follows the nearest fly).

### Fixed
- Two screenshots or GIFs saved in the same second no longer overwrite each other.
- Crash reports from the AppImage no longer try to write inside its read-only mount.

## 2.5.0 (2026-09-15)
- The brain panel follows the nearest fly; up to 16 flies.
