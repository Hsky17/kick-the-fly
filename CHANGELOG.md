# Changelog

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
