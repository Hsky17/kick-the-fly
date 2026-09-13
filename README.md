# Kick the Fly

A kick-the-buddy game where the buddy is a real fruit fly brain: the
**MaleCNS v1.0** connectome ([Google Research blog](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)),
all 166,700 neurons simulated live while you throw, flick, swat, bomb and torch it.

- **Hits fire real sensory neurons:**
  - head: head bristles and Johnston's organ
  - body: tactile neurons
  - legs: proprioceptive neurons
  - wings: wing sensory neurons
  - blowtorch: heat-sensing neurons
  - brake cleaner: smell and taste neurons (and it dissolves the fly)
- **Its reactions come from its descending neurons:** jumping, running, kicking, walking, backing up and turning.
- **Pain meter:** built from touch overload, heat-sensor activity and descending-neuron alarm. The **blowtorch** maxes it out.
- **Death and autopsy:** it can die. The autopsy compares every brain region's last 2 s alive with its calm baseline, and shows pain on a timeline.
- **Live brain view:** a front view of the brain built from the neurons' real cell-body positions. It lights up as the fly gets hurt. Press **B** for the big view.

![brain lighting up under the blowtorch](docs/brain.png)

![autopsy](docs/autopsy.png)

## Download and play (Windows)

**[Download KickTheFly.exe](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/latest/download/KickTheFly.exe)** (about 90 MB) and double-click it. You don't need to install anything, and the fly's whole brain is inside the exe.

- **Startup:** the first launch takes a few seconds while the exe unpacks.
- **Windows warning:** the exe isn't code-signed, so Windows SmartScreen may say "Windows protected your PC". Click **More info**, then **Run anyway**.
- **If it crashes:** it writes `KickTheFly-crash.txt` next to the exe.

## Run from source

Needs Python 3.11 and about 1.5 GB of disk for the connectome.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m connectome.loader build     # downloads the connectome (~1.1 GB) and builds data/graph.pkl
.venv\Scripts\python kick_the_fly.py                # the first run packs data/kick_brain.npz (~30 s)
```

To build the exe yourself (after one run from source, so the brain pack exists):

```powershell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

## Controls

| key | tool |
|---|---|
| 1 | hand: drag and throw |
| 2 | flick |
| 3 | fly swatter |
| 4 | bomb |
| 5 | blowtorch: hold to burn, pins pain at 100 |
| 6 | brake cleaner: hold to spray, dissolves the fly into a puddle |
| B | big live brain view (or click the brain panel) |
| R | new fly |
| Esc | quit |

## What is the connectome and what is a game rule

**Connectome**
- The spiking model (leaky integrate-and-fire over 10.5M signed synapses) and all the neuron firing.
- Which sensory neurons each hit drives.
- The descending neurons read out for reactions. The jump, run and kick groups are the DN types that responded most to head, body and leg touch when the sim was probed.

**Game rules**
- Which move each neuron group triggers.
- The ragdoll physics and standing back up.
- Jump direction, stun and damage.
- The pain index. The adult connectome has no neurons annotated as nociceptors, so pain is an estimate built from real signals, not a measurement of what the fly feels.
- Death. A sim can't die on its own, so on death its tonic drive is switched off and activity fades out.
- Brake cleaner dissolving the fly, and the brain slowing as it dissolves. Solvents depress nervous systems, so an inhibitory current grows on every neuron as the fly melts. How strong it is was picked for the game, not measured. The smell and taste neurons it fires are real.
- Fiber shapes in the brain view. Cell-body positions are real, but full neuron shapes aren't bundled, so each neuron is drawn from its cell body toward the center of its synaptic partners. Color is the fiber's direction: red left-right, green up-down, blue front-back.

The full mapping is in the docstring at the top of `kick_the_fly.py`.

## Credits

The connectome data is Janelia FlyEM MaleCNS v1.0, a collaboration between HHMI Janelia, the University of Cambridge, the MRC Laboratory of Molecular Biology and Google Research. It is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and available at [male-cns.janelia.org](https://male-cns.janelia.org/download/).

The exe bundles a compact pack derived from that data. The pack keeps the signed synapse counts, the neuron labels and the cell-body positions, and is otherwise unmodified.
