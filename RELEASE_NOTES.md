Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.

## For players

- **A real settings menu.** Press **Esc** for the pause menu: Resume, Challenges, Settings, Save State, Load State and Quit. Settings has tabs for Graphics, Audio, Brain, Controls and Accessibility, with explanations when you hover. Everything is saved, and all your old hotkeys still work (you can rebind them now too). Esc no longer quits straight away; Quit asks first.
- **Challenges.** Three games built on real fly experiments, with stars and best scores:
  - *Teach it to pick the right door:* choose which door zaps, train the fly, and watch it choose.
  - *How close can you sneak?:* creep up before its escape neuron spots you.
  - *Find its sweet tooth:* find the weakest sugar it still reaches for.
- **"Real flies do this too."** The first time the fly does something this game has checked against real fly science (dodging, reaching for sugar, its grooming neurons firing in the wind, avoiding a smell it learned to fear) a short card tells you. You can turn these off.
- **Slow motion and pause.** **Z** pauses time, **[** and **]** slow everything down to 0.1x, and **.** steps forward a tiny bit at a time.
- **Save states.** Save the exact moment, the fly's whole brain included, and load it later. Saves work between Windows and Linux.
- **Native Wayland on Linux.** The game runs natively on Wayland and falls back to X11 by itself; pick one in Settings if you need to.
- **Windows DPI fixes.** On 125% and 150% displays the game now draws at full resolution instead of looking blurry.
- **Accessibility.** Colorblind-safe and high-contrast brain colors, reduced flashing and larger text.
- **Windows players: your fly still remembers.** Training memory from earlier versions carries over from `Documents\Kick the Fly\memory` (including a Documents folder moved to OneDrive). Nothing is moved or deleted. The exe still isn't code-signed, so SmartScreen may warn you: click **More info**, then **Run anyway**.

## For researchers

- **Lab mode** (Esc > Mode): a validation dashboard, assays with repeated trials and statistics, live model parameters, recording and export, and protocol files. Reactions are tagged REAL (a descending-neuron readout) or RULE (a game rule).
- **Validation results** (held-out seeds 1000-1009, n = 10; pass = at least 1.5x and above a matched control, one-sided Wilcoxon p < 0.01; thresholds chosen for this release, not from the papers):

  | test | result |
  |---|---|
  | LPLC2/LC4 -> giant fiber DNp01 (von Reyn et al. 2014; Ache et al. 2019) | **PASS**: x11.8 vs x0.80 |
  | MDN -> backward walking (Bidaye et al. 2014) | **FAIL**: leg motor neurons x0.96 vs x0.96; doesn't reproduce |
  | sugar-pathway taste neurons -> MN9, not bitter (Shiu et al. 2024) | **PASS**: x2.11 vs x1.25 |
  | antennal JO-C/E -> aDN1/aDN2 grooming neurons (Hampel et al. 2015) | **PASS**: x4.87 vs x0.85 |
  | aDN1/aDN2 -> front-leg motor neurons (Hampel et al. 2015) | **FAIL**: x1.13, far below 1.5x |
  | T-maze conditioning, unpaired control (Tully & Quinn 1985) | **PASS**: PI 1.00 vs -0.03 (learning rule and choice are game rules; PI above real flies') |

  Sugar and bitter taste neurons aren't labeled by taste in MaleCNS v1.0, so those sets are chosen from wiring to the Yao & Scott 2022 sugar and bitter SEL neurons. Details and caveats are in the README's Validation section. No optomotor assay: pixel input through the photoreceptors doesn't carry a usable signal in this sim yet.
- **Assays and statistics:** T-maze (Tully & Quinn PI), looming escape (probability and latency vs approach speed) and MN9 sugar dose-response over any number of seeds, with mean ± 95% CI. Any surgery automatically gets unperturbed same-seed controls and a paired Wilcoxon test (paired t and Fisher's exact reported too).
- **Export:** spike times and firing rates for chosen neurons or regions to CSV and npz, with a metadata JSON (app version, seed, LIF parameters, thresholds, connectome version, brain pack checksum, surgery).
- **Protocols:** YAML files that define stimuli, recordings, flies, surgery or a standard assay. Run them from Lab > Protocols or headless.
- **Headless mode:** `KickTheFly --headless --protocol FILE --out DIR` and `--headless --validate` need no display (SSH, CI; the Windows exe from a console with `start /wait`). Lockstep runs are deterministic for a given seed on a given machine.
- **Tests:** `pytest` covers config, data paths and migration, save-state round trips, deterministic replay, protocols, the brain smoke test (DNp01 above baseline when LPLC2/LC4 are driven) and the full validation suite; the release workflow runs them on Ubuntu 22.04 and Windows before publishing.
