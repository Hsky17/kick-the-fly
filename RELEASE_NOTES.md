# Kick the Fly 2.8.0: bigger swarms, a faster simulation, real neuron shapes and video recording

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over.

## For players

- **Bigger swarms.** Press **N** for more flies, each running its own full 166,700-neuron brain. How many you can have
  now depends on how the simulation runs: up to 16 flies in the exe and AppImage (as before), and from source with Numba installed, one per CPU core up to 32 (24 on a 24-core PC). Each fly needs about 350 MB of free memory, and N tells you when there isn't enough. Press **F** to choose which fly the brain panel, surgery and
  training follow.
- **A faster simulation, from source.** If you run the game from source and install Numba (`pip install numba`), the
  brains run on compiled code that lets several flies think at the same time: 16 flies keep real time where plain NumPy manages 0.87x, and 32 run about twice as fast. In the game itself the drawing shares the work, so a big swarm still falls behind real time. PyTorch can run them on an
  NVIDIA or AMD graphics card (untested so far, see below). The exe and AppImage still use the standard engine, which
  hasn't changed speed. Your fly behaves exactly the same on every engine except the GPU ones.
- **Real neuron shapes.** Ten neurons in the brain view (the two giant fibers that trigger the escape jump, the two
  DNa02 steering neurons, and six from the learning centre) are now drawn from their real shapes, reconstructed from
  electron microscopy and downloaded from Janelia's neuPrint the first time you play online. Everything else is still
  an estimated fiber, and the view tells you which you're seeing.
- **Video recording.** **Shift+R** records a video of any length and Shift+R stops it: an MP4 if ffmpeg is installed,
  otherwise a GIF of up to a minute. It plays back at real speed, and it goes to your screenshots folder.
- **Fixes:** the fly's state (FLYING, STUCK ON FLYPAPER...) is back at the top left, the big brain view's buttons no
  longer draw over each other, and plain R resets the fly again while you're recording.
- **New screenshots** of everything in the game, including the open field, the orchard and the escape room.

## For researchers

- **Backend selection.** `--backend auto|cpu|numba|torch-cpu|torch-cuda|torch-rocm` (headless too, including
  `--validate`, whose worker processes inherit it), or Settings > Brain > Compute backend. `auto` prefers a GPU, then
  Numba, then NumPy; a backend that can't start falls back to NumPy and logs why. `--dtype float32|float64` sets the
  state precision (float32 default). Numba and PyTorch are optional source dependencies; the exe and AppImage include
  neither and don't load one you've installed.
- **Determinism guarantees and tolerances.**
  - `numba` and `torch-cpu` are **bit-exact** with the NumPy reference: same float operations, same order, same
    precision. `tests/test_backends.py` compares every spike, membrane potential and the adaptive gain over 1000 steps
    (5 s of brain time) in float32 and float64. The full validation suite (seeds 1000-1009) gives identical numbers on
    `cpu`, `numba` and `torch-cpu`, and matches the table in the README.
  - GPU backends are **not** bit-exact: a GPU sparse product may add a neuron's inputs in a different order, the last
    bit of a float32 sum then differs, and the network is chaotic, so individual spikes diverge within a few hundred
    steps. Their tolerance is statistical: brain-wide firing within 2% of NumPy's and per-population rates correlated at
    r > 0.95 over 1000 steps. That test exists but has **not yet run on a GPU** (below).
  - Every result records the backend and device that actually ran: validation JSON, benchmark JSON, exports and NWB
    files, save states and crash reports.
- **Benchmark reporting.** `--benchmark --backend NAME --flies 1 8 16 32 --seconds 5` reports paced and uncapped
  steps/s, sim/real ratio, neuron updates/s, synaptic events/s (measured spikes x the connectome's mean out-degree; it
  used to be an estimate from an assumed activity level) and resident memory, with the backend and device that ran.
  Measured (Core Ultra 7 270K Plus, 24 cores, Python 3.14; uncapped multiple of real time per brain): 1 brain 4.17x on NumPy vs 4.76x on Numba; 8 brains 2.21x vs 3.11x; 16 brains 0.91x vs 1.83x; 32 brains 0.38x vs 0.79x. `torch-cpu` 0.83x at 1 brain (it multiplies the whole matrix every step). Memory 310-390 MB per brain. Full tables and in-game numbers in the README's Performance section.
- **Citable.** `CITATION.cff` describes how to cite Kick the Fly and the MaleCNS v1.0 connectome it simulates (GitHub's
  "Cite this repository" button uses it). There's no DOI yet.

## What was tested on real hardware, and what wasn't

| | ran on real hardware? |
|---|---|
| `cpu` (NumPy) | Yes: Linux dev machine (Intel Core Ultra 7 270K Plus), and CI on Ubuntu 22.04 and Windows (full test and validation suite), which is also what the exe and AppImage run |
| `numba` | Yes, on the Linux dev machine only: bit-exactness tests, full validation suite, benchmarks, the 3D game with up to 23 flies. Not on Windows or macOS |
| `torch-cpu` | Yes, on the Linux dev machine only (PyTorch 2.14 CPU build): bit-exactness tests, full validation suite, benchmarks |
| `torch-cuda` | **No.** No NVIDIA GPU was available. The code path is the same as `torch-cpu` with a different device, but it has never run on a GPU |
| `torch-rocm` | **No.** The dev machine has an AMD Radeon RX 9070 XT, but no ROCm build of PyTorch was installed, so it wasn't run. Same caveat |

The GPU fly cap (32) and the GPU tolerance test in `tests/test_backends.py` are therefore untested. If you run a GPU backend, `python -m pytest tests/test_backends.py` and `--benchmark --backend torch-cuda` (or `torch-rocm`) will tell you whether it holds; reports welcome. The exe and AppImage were built and smoke-tested by CI (headless protocol run); the 3D game in the AppImage was not re-tested by hand for this release.

## Building from source

Nothing changed in how the brain pack is built. For the optional backends see README > Run from source > Optional:
faster simulation with Numba or PyTorch. `tools/make_screenshots.py` regenerates every README image, and
`tools/profile_sim.py` shows where a simulation step's time goes.
