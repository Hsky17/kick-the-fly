# Changelog

## Unreleased

### Added
- **Optogenetics Laser (Lab > Laser):** In-world aimable beam that activates or silences selected cell types directly in real time, replacing menu-only surgery for rapid targeted stimulation/silencing experiments.
- **Psychometrics Curve Generator (Lab > Psychometrics):** Sweeps stimulus parameters across continuous ranges, executes N trials with 95% confidence intervals, and exports publication-ready vector figures (PDF-1.4, SVG) and raw CSV data.
- **E-PG Compass Validation Test (Lab > Validation):** Negative validation assay testing whether ring attractor compass bumps form without visual landmark cues (documented negative validation result: MaleCNS v1.0 without landmark cues does not maintain a self-sustaining bump; no false compass HUD is displayed).
- **Classroom Mode & Lecture Protocols (Lab > Classroom):** Self-contained educational modules with guided steps, hypothesis prompts, and bundled interactive YAML lecture protocols (`protocols/lecture_*.yaml`).
- **Mystery Defect / Reverse Brain Surgery Challenge (Play mode):** Curated circuit silencing challenges without jargon where players test the fly with tools, request hints, and deduce missing neural circuits.
- **Predict-the-Neuron Minigame (Play mode):** Reaction minigame restricted strictly to descending motor readouts (jump, run, kick, back up, take off) where players predict upcoming movements from motor surge cues.
- **Escape-Room Arena (Play mode):** Multi-hazard emergent navigation gauntlet combining fan wind, flypaper strip, and hot lamp overhead; reach the sugar dish to stop the speedrun timer and generate a tamper-evident verification code (`KTF-<SEED>-<TIME>-<SIG>`).
- **Neural Clamp (Lab > Neural clamp):** Record reference spike trains from calibrated runs (looming, antennal grooming, sweet taste, calm baseline) and replay forced spikes into altered connectomes (lesion, threshold pruning, transmitter sign flips) to isolate structural changes from sensory feedback. Side-by-side activity diff table and export. Explicit dynamic clamp notice in UI and exports that forced spikes override membrane state and break closed-loop feedback.
- **Connectome Diff Mode (Lab > Connectome diff):** Run two flies (reference vs perturbed) side-by-side with identical seeds, initial states, and sensory inputs in lockstep. Real-time region-by-region activity divergence with autopsy-style diverging bar charts, timeline tracking the moment trajectories split, and CSV/JSON export.
- **Hemifield & Hemisphere Silencing:** Brain surgery presets for unilateral visual pathways (`LC10`, `LPLC2`, `LC4`, `LPTC`, `VS`, `HS`) and whole hemibrains. Validated behavioral consequences: failure to dodge blind-side looming, asymmetric steering bias (-1.8 Hz vs +0.10 Hz baseline), and broken 1v1 duel aiming on the blind side. Reported strictly as connectome wiring outcomes.
- **Global Inhibition Block / Picrotoxin (Lab > Robustness > Inhibition block):** 0-100% severity slider scaling inhibitory synaptic weights (`inhibition_scale = 1 - severity`). Emergent runaway excitation (>30 Hz brain-wide mean) without scripted seizures; before/after firing rate distributions; convulsion twitching and severity labels tagged as game rules.
- **Research Findings in Documentation:** Synapse threshold sweep (pruning <10 synapses drops 73.6% of connections while preserving all 4 validated behaviors), neurotransmitter sign flips (looming and grooming robust, sugar -> MN9 fails in 100% of trials), and looming critical path (LC4 -50%, LPLC2 -43%).

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
