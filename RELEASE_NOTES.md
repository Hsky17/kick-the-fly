Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.

## For players

- **Go outside.** Two new places to take the fly, in the 3D game (press **E**, or Settings > Brain > Arena):
  - **Open field:** grass, rocks and sky as far as you can see. A steady breeze blows through its real wind-sensing neurons, and the sun lights up its eyes. With open sky above, when it escapes it really goes: fly it out of sight and it's lost. Press **J** to call it back.
  - **Orchard:** a grove of fruit trees. The fly flies up to the fruit, lands and feeds, which lights up the same reward neurons sugar does and heals it. Fruit get eaten down, drop, and grow back a minute or so later. Some are fermented, with the same wobbly results as the alcohol drop. Spawn a few flies (**N**) and watch them jostle for fruit: nothing tells them to compete, they only notice each other the way they always have, by seeing each other loom and bumping into each other.
  - The room is still where you start, and the 2D game stays indoors.
- **New ways to play from the previous batch:** the *Mystery defect* challenge (one circuit is switched off; work out which), *Predict the move* (call the fly's next move from its descending neurons), and the *Escape room* speedrun with a verification code.
- **Fixes:** starting with `--autopilot` no longer crashes, and holding the laser near the fly in 3D no longer crashes the game.
- **Your fly still remembers,** your saves still load and your settings carry over. Saves now remember which arena you were in, and in the orchard which fruit were eaten.

## For researchers

- **Outdoor arenas, with their rules stated.** Wind drives the real JO-C/E neurons of each antenna and sunlight the photoreceptors; the split between the two sides by heading is a stated game rule, as is everything about the orchard (fruit, regrowth, the cap, the fly flying to fruit). Feeding drives the sugar-pathway taste and PAM reward neurons exactly as the sugar tool does. Feeds per fruit, regrow time and the cap are Lab parameters and protocol `params`, and `assay: orchard` runs the feeding schedule headless and reproducibly.
- **Know this before training in the orchard:** alcohol and fermented fruit smell through the real fermentation glomeruli DM1, DM2 and DP1m, and DM2 and DP1m are also two of the zapper's five scent glomeruli, so mushroom-body training on one partly generalises to the other.
- **E-PG compass, second test, negative.** Steady directional wind from the open field (the arena's own transduction, 8 directions, 10 held-out seeds, the visual test's criteria fixed in advance, nothing tuned) forms no head-direction bump: contrast 1.81x (1.71x without wind; 3.0x needed), persistence 101 ms (500 ms needed), direction tracking no better than shuffled directions (p = 0.17). No compass HUD ships.
- **Validation** (seeds 1000-1009): PASS looming -> giant fiber, sugar -> MN9, antennal JO-C/E -> aDN, T-maze conditioning. FAIL, reported as such: MDN -> backward walking, aDN -> front-leg motor neurons, E-PG bump from a driven wedge, E-PG bump from wind.
- **How robust are those results?** New Lab tools that change the connectome itself and re-run the behaviors with validation's own criteria:
  - *Synapse threshold:* dropping every connection below 10 synapses (73.6% of them, and every input of 10,151 neurons) leaves all four passing behaviors intact.
  - *Sign flips:* flipping a random half of the neurons whose transmitter prediction is under 70% confident breaks sugar -> MN9 in 3 of 3 trials; looming and antennal grooming survive.
  - *Critical path finder:* silence each candidate cell type and rank them; for looming, LC4 (-50%) and LPLC2 (-43%) carry nearly all of it.
  - *Inhibition block, neural clamp, connectome diff, hemifield lesions:* see the README's Lab section.
- **NWB export** of any recording (spikes, rates, stimuli, events, kinematics, surgery, arena, learned KC->MBON weights before and after, full metadata and citation), via optional `pynwb`.
- **Headless:** `--threshold-sweep`, `--signflip-test`, `--critical-path TARGET`, `--nwb`, `--arena NAME`.

## Building from source: the code moved

Everything now lives in a `kickthefly/` package (`core/`, `sim/`, `game/`, `ui/`, `lab/`, `data/`). `python kick_the_fly.py` still works exactly as before, and so does `python -m kickthefly`. Two commands changed: the connectome is built with `python -m kickthefly.sim.connectome.loader build` and the brain pack with `python -m kickthefly.sim.brainpack build`. Rebuild the brain pack once: it now carries each neuron's transmitter and its confidence (older packs still load; the sign-flip and inhibition tools then ask for a rebuild). An existing `data/graph.pkl` still loads. See `CONTRIBUTING.md` for where things go.
