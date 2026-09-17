# Kick the Fly

A kick-the-buddy game where the buddy is a real fruit fly brain: the
**MaleCNS v1.0** connectome ([Google Research blog](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)),
all 166,700 neurons simulated live while you throw, swat, bomb, burn, dissolve, zap, freeze and feed it to a spider. Or reward it with sugar.

**It's first person:** walk around a 3D living room and use your tools on the fly up close. Walk into it and you kick it. The original 2D version is still there with `--2d`.

**Play or Lab:** Play is the game, with challenges and scores. Lab mode adds research tools: a validation dashboard showing which published fly behaviors this simulation reproduces (and which it doesn't), repeated trials with statistics, data export and protocol files that also run headless.

![the 3D room](docs/room3d.png)

- **Hits fire real sensory neurons:**
  - head: head bristles and Johnston's organ
  - body: tactile neurons
  - legs: proprioceptive neurons
  - wings: wing sensory neurons
  - blowtorch: heat-sensing neurons
  - brake cleaner: smell and taste neurons (and it dissolves the fly)
  - freeze spray: cold-sensing neurons
  - zapper: every touch neuron plus a shock through its brain
  - spider bites: body and leg touch neurons
- **Its reactions come from its descending neurons:** running, kicking, walking, backing up and turning.
- **It can fly:** it takes off when its DNg02 wing-power neurons fire above normal, and flies away when its head-touch escape neurons fire.
- **Pain meter:** built from touch overload, heat and cold sensors, chemical senses and descending-neuron alarm. The **blowtorch** and **brake cleaner** max it out.
- **More pain neurons (P):** the wiring can't gain neurons, so the pain setting listens to more of the fly's real ones. **Normal** uses 9,080. **More** uses 11,392 and adds the rest of the body's sensory neurons. **Max** uses 13,238 and adds the ascending neurons that relay body signals to the brain. Higher settings also make each hit fire more of them.
- **Immortal mode (I):** it feels everything but can't die. It heals when you stop, and breaks out of spider silk.
- **Reward:** drop **sugar** and it walks over to eat. That lights up its PAM dopamine reward neurons and heals it. Its sugar-pathway taste neurons fire its proboscis motor neuron MN9, and when MN9 responds its proboscis comes out.
- **Alcohol (-):** drop a droplet of fermented fruit and the fly walks over and sips it. Drinking drives its real sweet taste pathway and its PAM dopamine reward neurons, the same ones sugar does, and its smell comes through the real fermentation glomeruli (DM1, DM2, DP1m). Getting drunk is a **game rule**: an inebriation level builds up with every sip and wears off over about 45 seconds, and while it lasts the game gives the fly tremors, a stumbling gait, wobbly flight and slower escape reflexes. No neuron in the simulation is actually intoxicated.
- **Death and autopsy:** it can die. The autopsy compares every brain region's last 2 s alive with its calm baseline, and shows pain on a timeline.
- **It sees you coming:** move a weapon at it fast and its real looming detectors (LPLC2 and LC4) fire its giant fiber escape neuron, so it dodges. Sneak up slowly and it won't notice.
- **Brain surgery (O):** silence or stimulate real neuron groups and watch what happens. Switch on the moonwalker neurons and it backs up; silence the giant fiber and it can't dodge.
- **Neuron inspector:** in the big brain view, click any neuron to see its type, how fast it's firing, and its strongest connections in the connectome.
- **1v1 duel (X):** the fly gets a blaster and can kill you, and every part of the fight runs through its brain:
  - it sees you through its real target-tracking neurons (LC10), which steer it toward you through its steering neurons (DNa02);
  - it shoots when its small-object detectors fire its DNp35 neurons;
  - landing a hit fires its reward dopamine neurons, so it learns to like hunting you;
  - hurting it fires its punishment dopamine neurons, so it learns to fear you, and it runs away and stops shooting.

  Silence its tracking neurons in brain surgery and it can't aim.
- **Multiple flies (N):** press N to spawn another fly, up to 16 at once, each running its own complete, independent connectome — 166,700 neurons apiece. They notice each other for real: a fly closing in fast fires another's actual looming detectors (LPLC2/LC4) and makes it dodge, and bumping into each other fires real touch neurons. The brain panel, training and surgery follow the fly nearest to you; R goes back to one fly. Only the original fly's mushroom-body learning is saved between sessions. Every fly is its own brain thread, so with many flies the brains can fall behind real time (see Performance).
- **Real training (T):** the fly learns with its actual mushroom body. Pair a smell with a shock or with sugar and dopamine weakens the real Kenyon cell to output neuron synapses for that smell, just like in real flies. The Training panel runs lab-style conditioning and graphs the learning curve. Memory is saved between flies and sessions (see File locations). Hurting the fly while it smells a tool trains it too.
- **Arenas (E):**
  - **fan:** wind that fires its wind-sensing neurons, which excite its antennal grooming command neurons
  - **flypaper:** it gets stuck and struggles
  - **pool:** it floats, gets wet wings, and can drown
  - **lamp:** it's drawn to the light and singes itself on the bulb
- **Sound:** every sound is generated in code. The wing buzz follows its flight neurons. **M** mutes.
- **Save and share:** **F12** (3D) or **S** (2D) saves a screenshot and **G** saves a GIF of the last 6 seconds. **L** toggles time-lapse frame recording (2x, 5x, 10x, 20x speed-up exported to MP4 via ffmpeg or animated GIF; tagged as GAME RULE: visual recording). The autopsy can save a GIF of the death.
- **Slow motion and save states:** pause time, slow everything to 0.1x, step it 1/60 s at a time, and save or load the whole simulation (see Time controls).
- **Live brain view:** a front view of the brain built from the neurons' real cell-body positions, shaded by depth. Pain-sensing neurons glow orange and everything else glows cyan when firing (blue/yellow and high-contrast palettes in Settings > Accessibility). Press **B** for the big view.

![swatting in first person](docs/swat3d.png)

![stuck on flypaper](docs/flypaper3d.png)

![see-through brain panel](docs/see-through.png)

![brain lighting up under the blowtorch](docs/brain.png)

![spider wrapping the fly](docs/spider.png)

![1v1 duel](docs/duel.png)

![training](docs/training.png)

![neuron inspector](docs/inspect.png)

![brain surgery](docs/surgery.png)

![the lamp arena](docs/lamp.png)

![autopsy](docs/autopsy.png)

## Download and play (Windows)

**[Download KickTheFly.exe](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/latest/download/KickTheFly.exe)** (about 90 MB) and double-click it. You don't need to install anything, and the fly's whole brain is inside the exe.

- **Startup:** the first launch takes a few seconds while the exe unpacks.
- **Windows warning:** the exe isn't code-signed, so Windows SmartScreen may say "Windows protected your PC". Click **More info**, then **Run anyway**.
- **Your fly remembers:** training memory from earlier versions is still read from `Documents\Kick the Fly\memory`, including a Documents folder moved to OneDrive. Nothing is moved or deleted.
- **Fullscreen and sharp scaling:** press **F11**, or start it with `KickTheFly.exe --fullscreen`. The game is DPI aware, so 125% and 150% displays draw at full resolution instead of blurry.
- **Headless:** `start /wait KickTheFly.exe --headless --protocol smoke.yaml --out results` runs without a window (bundled protocols can be named without a folder; see Lab tools).
- **If it crashes:** it writes `KickTheFly-crash.txt` next to the exe and in `%LOCALAPPDATA%\Kick the Fly`.
- **Checksums:** each release has `SHA256SUMS`; `Get-FileHash KickTheFly.exe` should match.

## Download and play (Linux)

**[Download KickTheFly-x86_64.AppImage](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/latest/download/KickTheFly-x86_64.AppImage)** (about 110 MB), `chmod +x` it, and run it. No install needed, and the fly's whole brain is inside it. It's built on Ubuntu 22.04, so it runs on most distros from then on.

- **Startup:** the first launch takes a few seconds while it unpacks.
- **Arch Linux:** an AUR package `kickthefly-bin` is in `packaging/aur/` (installs the release AppImage and a menu entry).
- **Wayland and X11:** it uses native Wayland when `WAYLAND_DISPLAY` is set and falls back to X11/XWayland by itself if that fails. Force one with `--backend wayland` or `--backend x11`, or in Settings > Graphics (applies on restart). An `SDL_VIDEODRIVER` you set yourself always wins. Mouse look uses relative pointer mode on both.
- **GPU:** needs OpenGL 3.3 for the 3D room. Without it the game logs why and starts the 2D game (`--2d` skips the check). Software rendering (`LIBGL_ALWAYS_SOFTWARE=1`) works but is slow.
- **If it won't start (FUSE):** AppImages mount themselves with FUSE. Without FUSE (no `libfuse2`/`fusermount`, containers, some minimal distros) run `./KickTheFly-x86_64.AppImage --appimage-extract-and-run`, or set `APPIMAGE_EXTRACT_AND_RUN=1`. `sudo apt install libfuse2` fixes it on Debian/Ubuntu.
- **Headless over SSH:** `./KickTheFly-x86_64.AppImage --headless --protocol smoke.yaml` works with no `DISPLAY` or `WAYLAND_DISPLAY`.
- **Sound:** plays through PulseAudio (including PipeWire's PulseAudio server) or ALSA.
- **If it crashes:** it writes `KickTheFly-crash.txt` next to the AppImage and in `~/.local/state/kickthefly/`.

## File locations

| | Windows | Linux |
|---|---|---|
| settings (`config.toml`) | `%APPDATA%\Kick the Fly` | `$XDG_CONFIG_HOME/kickthefly` (`~/.config/kickthefly`) |
| training memory | `Documents\Kick the Fly\memory` | `$XDG_DATA_HOME/kickthefly/memory` (`~/.local/share/kickthefly/memory`) |
| save states | `Documents\Kick the Fly\saves` | `~/.local/share/kickthefly/saves` |
| exports, scores, validation runs | `Documents\Kick the Fly` | `~/.local/share/kickthefly` |
| your protocol files | `Documents\Kick the Fly\protocols` | `~/.local/share/kickthefly/protocols` |
| screenshots and GIFs | `Pictures\Kick the Fly` | `<xdg-user-dir PICTURES>/Kick the Fly` (`~/Pictures/Kick the Fly`) |
| crash reports and log | `%LOCALAPPDATA%\Kick the Fly` | `$XDG_STATE_HOME/kickthefly` (`~/.local/state/kickthefly`) |

Documents and Pictures on Windows come from the Known Folders API, so redirected and OneDrive folders work. On Linux, versions before 2.6 used `~/Documents/Kick the Fly/memory` and `~/Pictures/Kick the Fly`; on first launch the memory and any screenshots are copied to the new locations and the originals are left alone. A broken or missing `config.toml` falls back to default settings with a warning (a broken one is kept as `config.toml.bad`). `KICK_THE_FLY_HOME=/some/folder` keeps everything in one folder (portable use, tests).

## Repo layout

```
kick_the_fly.py            launcher shim: `python kick_the_fly.py ...` works exactly as before
kickthefly/                the package everything lives in (`python -m kickthefly` runs the same game)
  core/                    clock, save states, settings, user folders, crash reports, version
  sim/                     the connectome: loader, brain pack, the LIF simulator
  game/                    the 2D game, the 3D room, physics, tools, arenas
  ui/                      menu framework and settings screens
  lab/                     validation, assays, challenges, statistics, protocols, recording and export, Lab tools
  data/                    non-code assets bundled inside the package
tests/  protocols/  docs/  packaging/  tools/
data/                      not in git: the connectome download, graph.pkl and the brain pack
```

The canonical map of what comes from the connectome and what is a game rule is the module docstring at the top of
`kickthefly/game/kick_the_fly.py`; the shim in the repo root points at it. `CONTRIBUTING.md` says where new code
goes. This layout arrived in 2.7; nothing user-facing moved, so existing saves, training memory, settings, protocol
files and command lines are unchanged.

## Run from source

Needs Python 3.11, a GPU with OpenGL 3.3 for 3D, and about 1.5 GB of disk for the connectome.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m kickthefly.sim.connectome.loader build   # downloads the connectome (~1.1 GB) and builds data/graph.pkl
.venv\Scripts\python kick_the_fly.py                # the first run packs data/kick_brain.npz (~30 s)
```

On Linux/macOS, use `python3` and forward slashes instead:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m kickthefly.sim.connectome.loader build   # downloads the connectome (~1.1 GB) and builds data/graph.pkl
.venv/bin/python kick_the_fly.py                    # the first run packs data/kick_brain.npz (~30 s)
```

Tests (the validation suite takes a few minutes; `-m "not validation"` skips it):

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest
```

To build the exe yourself (after one run from source, so the brain pack exists; add `data/validation_results.json` from `--validate` for the real-science popups):

```powershell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

To build the AppImage yourself (same prerequisite):

```bash
./build_appimage.sh
```

Releases are built by `.github/workflows/release.yml` on a tag push: the brain pack is built from the public connectome, all tests (including validation) run on Ubuntu 22.04 and Windows, the AppImage is built on Ubuntu 22.04 and the exe on Windows, and the release is published with `SHA256SUMS` only if everything passes.

## Controls

Every key below can be rebound in Settings > Controls (a key that's already taken swaps with that action). Esc and the tool keys 0-9 are fixed.

| key | what it does |
|---|---|
| Esc | close a panel, or open the pause menu: Resume, Challenges (Play) or Lab tools (Lab), Settings, Save State, Load State, Mode, Quit |
| WASD | walk (Shift sprint, Ctrl or C crouch); walk into the fly to kick it |
| Mouse | look around; left click uses the tool in your hand |
| 1-9, 0, - or mouse wheel | pick a tool: hand, flick, swatter, bomb, blowtorch, brake cleaner, zapper, freeze spray, spider, sugar, alcohol |
| Tab | free the mouse to click the brain panel and menus (click the room to look again) |
| B | big live brain view; click a neuron to inspect it |
| O | brain surgery |
| T | training: teach it to fear or like a smell (saved between sessions) |
| X | 1v1 duel: the fly gets a blaster and can kill you (R respawns you) |
| E | arena: room, fan, flypaper, pool, lamp |
| P / I | pain neurons / immortal mode |
| K | brain stethoscope (spike sonification clicks in big brain view / body parts) |
| L | time-lapse record (2x-20x speedup to MP4/GIF; toggle on/off) |
| Y | autopilot / spectator mode (hands-off orbit camera) |
| F10 | photo mode / free camera with depth of field |
| M | mute |
| F12 (S in 2D) / G | save a screenshot / a GIF of the last 6 seconds |
| V | brain panel: solid, see-through, faint, hidden (hidden gives the room the whole screen) |
| U | menu size: crisp (sharp whole-pixel scaling, the default) or large |
| F11 or Alt+Enter | fullscreen; the game fills any screen with no black bars |
| N | spawn another fly (up to 16), each running its own independent brain |
| R | reset to a single fresh fly |
| Z | pause or resume time |
| [ / ] | slower / faster: 0.1x, 0.25x, 0.5x, 1x |
| . | single step while paused (1/60 s of the room and the matching brain steps) |
| H | controls help |

Command line: `--2d`, `--fullscreen`, `--backend wayland|x11`, `--seed N`, and for headless runs `--headless`, `--validate`, `--protocol FILE`, `--out PATH`, `--workers N`, `--seeds 1000-1009`, `--strict`.

## Settings

Esc > Settings. Changes apply right away and are saved to `config.toml`; hover any setting for a plain explanation.

- **Graphics:** fullscreen, resolution scale (3D drawn smaller and stretched, for weak GPUs), FPS cap, VSync (restart), display backend (Linux only, restart), brain panel style, menu size, UI scale.
- **Audio:** master, wing buzz and sound effects volume, brain stethoscope (spike sonification clicks, hotkey K), mute.
- **Brain:** Play/Lab mode, pain neurons, immortal, sim speed, random seed (applies on R), real vs rule tags, real-science popups. Brain settings are tagged **Connectome** (changes how the simulation runs) or **Game rule** (a rule the game adds on top).
- **Controls:** mouse sensitivity, invert Y, field of view, key bindings.
- **Accessibility:** colorblind-safe brain view colors (blue/yellow) and a high-contrast palette, reduced flashing (no screen shake, flashes, sparkles, scanning band or blinking), larger text.

## Play and Lab

**Play** (the default) is the game plus three **challenges** in the pause menu, each built on a real experiment:

- **Teach it to pick the right door:** pick which of two smelly doors zaps. The fly is trained on its real mushroom body, then chooses a door 10 times. Score: right choices. The practice memory is put back afterwards, so it never changes how the fly treats your tools.
- **How close can you sneak?:** creep up on the fly. When its giant fiber fires it dodges. Score: how close you got, in fly lengths.
- **Find its sweet tooth:** offer sugar at different strengths and find the weakest one its proboscis motor neuron still responds to, in 8 tries.

In Play mode a short **"Real flies do this too"** card appears the first time the fly does something that passed this game's validation (dodging, reaching for sugar with its proboscis, its antennal grooming neurons firing in the fan's wind, avoiding a smell it learned to fear). Behaviors that failed validation never get one. Turn the cards off in Settings > Brain.

**Lab** (Esc > Mode, or Settings > Brain) replaces Challenges with **Lab tools**:

- **Validation:** every test with PASS or FAIL, the measured numbers, the pass criteria and the citation. "Run validation now" reruns the suite on your PC.
- **Assays and repeated trials:** T-maze conditioning (Tully & Quinn performance index), looming escape (escape probability, latency and distance vs approach speed) and sugar response (MN9 dose-response) over any number of flies (seeds), with mean and 95% confidence interval. Pick a surgery and every fly also runs unperturbed with the same seed as its control, compared with a paired Wilcoxon signed-rank test (paired t-test and, for yes/no outcomes, Fisher's exact test alongside). Each fly is a fresh, untrained brain in a worker process; your saved training memory isn't touched. Results export to JSON and CSV.
- **Parameters:** the LIF model's parameters (noise, tonic drive, target rate, sensory gain, gain adaptation; tagged MODEL) and the game-rule thresholds that turn neuron firing into moves, live. Validation results, exports and save states record when anything is changed from the defaults.
- **Record and export:** pick neuron groups and a duration and record the fly you're looking at while you play: spike times and firing rates as CSV and npz, with a metadata JSON (app version, seed, parameters, thresholds, connectome version, brain pack checksum, surgery).
- **Simulation benchmark:** measures simulation throughput across 1, 8, and 16 flies: paced real-time ratio, uncapped steps/s, neurons/s, synapse updates/s, and memory footprint.
- **Protocols:** YAML experiment files, from the bundled examples or your protocols folder.
- **Real vs rule tags** are on by default in Lab: each reaction in the brain panel and its popup is tagged REAL (live descending-neuron firing crossed a threshold; the movement itself is always game physics) or RULE (a game rule).

## Lab tools: protocols and headless runs

A protocol is either a stimulus schedule with recordings or a standard assay over many flies:

```yaml
name: looming-giant-fiber
seed: 100
flies: 5
warmup_s: 2
duration_s: 3
surgery: {"type:LPLC2,LC4": -1}     # silence (-1) or stimulate (1); each seed also runs unperturbed as its control
stimuli:
  - {at_s: 1.0, for_s: 1.0, target: loom, strength: 0.9, recruit: 0.6}      # game-style stimulus
  - {at_s: 2.5, for_s: 0.5, target: "type:MDN", mode: drive, amp: 0.5}      # constant activation
recordings:
  - {name: giant_fiber, neurons: dnp01}
  - {name: descending, neurons: "superclass:descending_neuron"}
```

```yaml
name: kc-silencing-tmaze
assay: tmaze          # tmaze | looming | sugar
seed: 3000
flies: 6
surgery: {"prefix:KC": -1}
```

Neurons are named by a group (`loom`, `escape`, `head`, `reward`, `sweet`, `dnp01`, `mn9`, `adn`, `jo_ce`, `mn_front`...), `type:A,B`, `prefix:KC`, `superclass:descending_neuron` or `rows:1,2,3`. The full format is in `kickthefly/lab/protocol.py`, and examples are in `protocols/` (bundled in the exe and AppImage: `--protocol smoke.yaml`, `looming-giant-fiber.yaml`, `kc-silencing-tmaze.yaml`, `sugar-dose-response.yaml`).

Run one without a window, from the exe, the AppImage or source:

```bash
./KickTheFly-x86_64.AppImage --headless --protocol protocols/looming-giant-fiber.yaml --out results
python kick_the_fly.py --headless --validate --out validation.json --strict
python kick_the_fly.py --benchmark --flies 1 8 16 --seconds 5
```

Headless runs need no display (SSH, CI), never open a window or audio device, write their files to `--out` (or the exports folder), and exit 0 on success, 2 for a bad or missing file, and with `--validate --strict` 1 if a validation result differs from the expected one. On Windows use `start /wait KickTheFly.exe ...` from cmd. Runs are seeded and stepped in lockstep, so the same protocol and seed give the same spikes on the same machine.

## Time controls and save states

- **Pause (Z), slow motion ([ and ]) and single step (.):** the room and every brain slow down together, so spikes and the reactions they cause stay lined up. An on-screen badge shows the state. You can still look and walk around at full speed.
- **Save State / Load State** (pause menu) saves the whole simulation: every neuron's membrane potential and refractory state, synaptic gain, the random generators, the learned Kenyon cell to MBON weights, surgery, Lab parameters, the arena, every fly's body and timers, sugar piles, the seed and (3D) you. The `.ktfsave` format is versioned and platform independent, so a save made on Linux loads on Windows and the other way round. Saves from a newer version, from the other (2D/3D) game or from a different brain pack are refused with a reason. Things in flight (bombs, sprays, the spider) aren't saved.
- **Deterministic runs:** with the same seed and the same inputs a lockstep run (headless, protocols, validation, the tests) replays spike for spike. The live game runs each brain on its own real-time thread, so play itself isn't bit-for-bit repeatable.

## Validation

`kickthefly/lab/validation.py` asks whether this simulation reproduces published results, and reports pass or fail with numbers. Every cell type was checked against the MaleCNS v1.0 annotations, whose synonyms record the published names (DNg62 and DNge078 are "Hampel 2015: aDN1/aDN2", GNG540/GNG550 are "Yao & Scott 2022: Sugar SEL PN", DNg28 is "Yao & Scott 2022: Bitter-SEL").

Method: seeds 1000-1009, never used while developing (exploratory probing used seeds 0-299). A set of neurons is driven for 2 s after 2 s of calm, from the same brain snapshot as a matched control set of the same size. **Pass: the readout's driven/baseline ratio averages at least 1.5x and beats the control's in a one-sided Wilcoxon signed-rank test, p < 0.01.** For conditioning: PI at least 0.5, the unpaired control's |PI| at most 0.25, and paired above unpaired (p < 0.01). These thresholds were chosen for this release after exploratory probing, not taken from the papers: a pass means the sim shows the effect in the stated direction and strength, not that its numbers match the papers'.

Results of this release (n = 10 flies, mean ± SD):

| test | readout: drive vs control | result |
|---|---|---|
| Looming detectors LPLC2 + LC4 excite the giant fiber DNp01 ([von Reyn et al. 2014](https://www.nature.com/articles/nn.3741); [Ache et al. 2019](https://www.cell.com/current-biology/fulltext/S0960-9822(19)30138-1)) | DNp01 x11.81 ± 1.77 vs x0.80 ± 0.18 for 311 random visual projection neurons, p < 0.001 | **PASS** |
| MDN activation drives backward walking ([Bidaye et al. 2014](https://pubmed.ncbi.nlm.nih.gov/24700860/)) | leg motor neurons x0.96 ± 0.03 vs x0.96 ± 0.05 for 4 random descending neurons, p = 0.38 | **FAIL: does not reproduce** |
| Sugar-sensing taste neurons activate the proboscis motor neuron MN9; bitter ones don't ([Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9)) | MN9 x2.11 ± 0.40 vs x1.25 ± 0.29 for bitter-pathway neurons, p < 0.001 | **PASS** |
| Antennal mechanosensory neurons JO-C/E excite the antennal grooming neurons aDN1/aDN2 ([Hampel et al. 2015](https://elifesciences.org/articles/08758); Shiu et al. 2024) | aDN1/aDN2 x4.87 ± 1.70 vs x0.85 ± 0.30 for random sensory neurons, p < 0.001 | **PASS** |
| aDN1/aDN2 activation drives antennal grooming, a front-leg movement (Hampel et al. 2015) | front-leg motor neurons x1.13 ± 0.07 vs x0.95 ± 0.06, p < 0.001 | **FAIL: too weak** (consistent, but a 13% rise is far below the 1.5x bar) |
| Odor + shock conditioning gives a positive T-maze performance index; unpaired doesn't ([Tully & Quinn 1985](https://pubmed.ncbi.nlm.nih.gov/3939242/)) | PI 1.00 ± 0.00 vs unpaired -0.03 ± 0.15, p < 0.001 | **PASS** (with caveats below) |

What the failures and passes mean:

- **MDN:** in this sim MDN activity doesn't reach the leg motor neurons at all. The game's "backs up" reaction reads MDN directly, which is a game rule, and gets no real-science card.
- **aDN to front legs:** the upstream half of the grooming circuit (antennal touch to aDN) reproduces strongly; the motor half doesn't. The fly shows no grooming movement, and the GROOM reaction only logs the command neurons.
- **Sugar:** the dataset doesn't label taste neurons by taste, so the sugar and bitter sets are chosen from their wiring to the Yao & Scott 2022 sugar and bitter neurons (MN9 is never used to choose them). Driving 30 random head taste neurons also raises MN9 somewhat.
- **T-maze:** the learning rule, shock driving dopamine neurons and the choice at the T-maze are game rules running on the connectome's real synapses; the test shows they give odor-specific memory. The PI of 1.00 is above real flies' typical ~0.8-0.9 and not tuned to match. The approach output neurons' overall firing barely differs between the two odors (30.9 vs 31.0 spikes/s), so the choice is read from the learned synapses, not from output-neuron firing.
- **Not tested:** optomotor responses. Pixel input through the photoreceptors didn't carry a usable signal in this sim, so there is no honest way to ground one yet.

`pytest` runs the suite and fails if any result changes in either direction.

## Performance

Measured on an AMD Radeon RX 9070 XT / 24-thread CPU, Python 3.11 (`tools/bench_sim.py`, and the 3D game with flies spawned):

| | before (2.5.0) | after (2.6.0) |
|---|---|---|
| 1 brain, paced / uncapped | 1.00x real time / 4.17x | 1.00x / 4.21x |
| 8 brains, no game loop, paced / uncapped | 1.00x / 2.16x | 1.00x / 2.21x |
| 3D game, 1 fly | 200 steps/s (1.00x), 62 fps | 200 steps/s (1.00x), 62 fps |
| 3D game, 8 flies | 0.41x real time, 60 fps | 0.43x real time, 59 fps |

With several flies in the game, the brain threads, the renderer and the brain view share Python's interpreter lock, so the brains fall behind real time. That was already true before 2.6 and isn't changed by it.

## What is the connectome and what is a game rule

**Connectome**
- The spiking model (leaky integrate-and-fire over 10.5M signed synapses) and all the neuron firing.
- Which sensory neurons each hit drives.
- The descending neurons read out for reactions. The jump, run and kick groups are the DN types that responded most to head, body and leg touch when the sim was probed.
- Take-off and flight speed come from DNg02, the wing-power descending neurons.
- In the 1v1 duel: aiming comes from the steering neurons DNa02 and DNa01 (right minus left), shooting from DNp35 and DNpe052, and walking toward you from DNp09. The pathways from target-tracking LC10 to DNa02 and from the small-object detectors LC11/18/21/26 to DNp35 are the connectome's own wiring. In testing, driving LC10 on one side took that side's DNa02 from about 1 to about 19 spikes/s. Whether it fights or flees is read from its mushroom body synapses for your smell.
- Dodging: the looming detectors LPLC2 and LC4 exciting the giant fiber DNp01 is the connectome's own wiring (validated: x11.8 vs x0.8 for a control).
- The wind, humidity and light neurons each arena fires, and the Kenyon cell patterns each tool's scent produces.
- The REWARD meter reads the PAM dopaminergic neurons.
- Drinking alcohol drives the same real pathways sugar does (the sugar-pathway taste neurons and the PAM reward neurons), and the droplet's smell drives the real olfactory neurons of the fermentation glomeruli DM1, DM2 and DP1m.
- Antennal wind excites the antennal grooming command neurons aDN1/aDN2 (validated); the GROOM reaction reads them.
- Sugar reaching the proboscis motor neuron MN9 (validated); the PROBOSCIS reaction reads MN9 during an eating bout. Which taste neurons count as sugar-pathway ones is chosen from the connectome's wiring to the annotated sugar and bitter SEL neurons.
- Everything inside the assays and protocols: how drive spreads, which neurons respond, and what silencing a group does.

**Game rules**
- Which move each neuron group triggers, and the thresholds (all adjustable in Lab > Parameters).
- The ragdoll physics and standing back up.
- Jump direction, stun and damage.
- The pain index. The adult connectome has no neurons annotated as nociceptors, so pain is an estimate built from real signals, not a measurement of what the fly feels.
- Death. A sim can't die on its own, so on death its tonic drive is switched off and activity fades out.
- Brake cleaner dissolving the fly, and the brain slowing as it dissolves. Solvents depress nervous systems, so an inhibitory current grows on every neuron as the fly melts. How strong it is was picked for the game, not measured. The smell and taste neurons it fires are real.
- Freezing and spider venom damping the brain, and the zapper's shock going into a random 30% of neurons (the sim has no current path to place it).
- Alcohol inebriation. Drinking raises a scripted inebriation level (0 to 1, decaying over ~45 s) that the game turns into tremors, a stumbling gait, wobbly flight and delayed escape reflexes. Ethanol's real pharmacology is not modelled: the simulated neurons are unaffected, and only the body's movement is degraded. The Lab mode Model Assumptions page lists this too.
- Sugar switching on the PAM reward neurons directly. In this sim taste input alone doesn't reach them, so sugar drives them the way PAM activation experiments do. The fly walking to the sugar and the proboscis coming out are also game rules (what triggers the proboscis is MN9).
- How looming reaches the fly. The game measures how fast an object grows in its view and drives LPLC2/LC4 directly. Streaming pixels through the sim's own photoreceptors didn't work: the looming signal stayed inside the brain's random flicker. The same transduction runs in the looming assay, so part of its speed dependence is this rule, not a measurement.
- Learning. The plasticity happens on the connectome's own synapses: all 41,495 Kenyon cell to MBON connections that dopamine neurons reach. Which dopamine neurons gate which output neurons comes from the connectome's 37,909 dopamine to output neuron synapses: PPL1 punishment dopamine for MBON11-20 and 30-35, and PAM reward dopamine for MBON01-10, 21, 24 and 26-29. That matches the published map. The rule is the one found in real flies: dopamine plus Kenyon cell activity weakens the synapse. Game rules: pain driving PPL1, sugar driving PAM, each tool having a smell, and the learning rate and forgetting speed. In testing, 10 pairings raised fear of the trained smell from 0 to 0.65 while an untrained smell stayed at 0.01, and the trained smell's approach output neurons dropped from 32 to 30 spikes/s.
- The T-maze (challenge and assay): each odor being a fixed set of 6 glomeruli, the shock driving PPL1 and leg touch neurons, and the choice at the fork. The fly smells each arm and picks the one whose learned drive (liking minus fear, read from its synapses) is higher, plus decision noise. The performance index is computed as in Tully & Quinn 1985.
- The looming assay's escape rule (DNp01 above its threshold before contact) and the sneak challenge's scoring.
- The sugar assay's dose (the share of sugar-pathway taste neurons driven) and what counts as a proboscis extension (MN9 at least 1.5x its rate just before).
- Real-science cards: which reactions get one is decided by the validation results, and only passing tests count.
- Slow motion: the room's physics still advances in 1/60 s ticks; bodies are drawn in between.
- Being drawn to the lamp, and the arena physics.
- In the 1v1 duel:
  - that it has a blaster at all;
  - where you appear in its view, which the game computes;
  - the gun's automatic up/down aim toward your chest;
  - hits firing its reward dopamine neurons, and getting hurt near you firing its punishment ones;
  - how learned fear switches its attention off you;
  - reversal learning: new opposite dopamine restores that smell's weakened synapses in the other compartment, so fear can overturn liking.

  The overall fly-brain firing of those output neurons was too noisy in this sim to read a decision from, so the choice is read from the learned synapses themselves.
- The flight path.
- Everything about the 3D room: the fly's 3D body, physics, walking and flight, and your tools. They use the 2D game's tuned physics scaled to meters, so the brain gets the same kinds of hits as before. What triggers take-off is from the neurons, but where it flies is not.
- Fiber shapes in the brain view. Cell-body positions are real, but full neuron shapes aren't bundled, so each neuron is drawn from its cell body toward the center of its synaptic partners. Color is the fiber's direction: red left-right, green up-down, blue front-back.
- Bilateral symmetry and mirror-averaging. In the raw connectome, bilateral asymmetries arise from both true biology and uneven EM reconstruction/proofreading depth between hemispheres, producing a small spontaneous turning bias in quiet walking (~+0.10 Hz DNa steering bias). The headless audit command (`--audit-asymmetry`) and the Lab Asymmetry page measure L vs R synapse counts and firing rates for key cell types (DNa01, DNa02, LC10, LPLC2, LC4, DNp01). An optional setting (`brain.mirror_weights` or `--mirror-weights`) averages synaptic weights across 77,507 paired bilateral neurons ($W_{sym} = 0.5(W + P W P^T)$). Because this modifies the raw connectome dataset, it is tagged strictly as a Game Rule.
- Brain stethoscope (spike sonification). Synthetic audio clicks triggered when neurons spike in a user-probed neuropil region (mushroom body, antennal lobe, central complex, optic lobes, motor neurons) or inspected neuron group. Hotkey K or button in the big brain view. Tagged strictly as a Game Rule: this is synthetic audio sonification for intuitive listening, not a biophysical local field potential (LFP) or extracellular microelectrode recording.

The full mapping is in the docstring at the top of `kickthefly/game/kick_the_fly.py` (the `kick_the_fly.py` shim in the repo root points at it), and per assay in `kickthefly/lab/assays.py`.

## Credits

The connectome data is Janelia FlyEM MaleCNS v1.0, a collaboration between HHMI Janelia, the University of Cambridge, the MRC Laboratory of Molecular Biology and Google Research. It is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and available at [male-cns.janelia.org](https://male-cns.janelia.org/download/).

The exe and AppImage bundle a compact pack derived from that data. The pack keeps the signed synapse counts, the neuron labels (type, superclass, subclass, instance), body IDs and the cell-body positions, and is otherwise unmodified.
