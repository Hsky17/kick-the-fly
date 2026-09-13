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
- **Its reactions come from its descending neurons:** jumping, running, kicking, walking, backing up and turning.
- **Pain meter:** built from touch overload, heat-sensor activity and descending-neuron alarm. The **blowtorch** maxes it out.
- **Death and autopsy:** it can die. The autopsy compares every brain region's last 2 s alive with its calm baseline, and shows pain on a timeline.

![autopsy](docs/autopsy.png)

## Run

Needs Python 3.11 and about 1.5 GB of disk for the connectome.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m connectome.loader build     # downloads the connectome (~1.1 GB) and builds data/graph.pkl
.venv\Scripts\python -m connectome.layout build     # brain map layout for the side panel (~2 min)
.venv\Scripts\python kick_the_fly.py
```

## Controls

| key | tool |
|---|---|
| 1 | hand: drag and throw |
| 2 | flick |
| 3 | fly swatter |
| 4 | bomb |
| 5 | blowtorch: hold to burn, pins pain at 100 |
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

The full mapping is in the docstring at the top of `kick_the_fly.py`.

Connectome data: Janelia FlyEM MaleCNS v1.0, downloaded from `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`.
