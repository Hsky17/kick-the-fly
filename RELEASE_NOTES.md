# Kick the Fly 2.8.1: it finally looks like a fruit fly

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over. Everything in
[2.8.0](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/tag/v2.8.0) (bigger swarms, the faster Numba
and PyTorch backends, real neuron shapes, video recording) is in this build too.

## For players

The fly looked like a bee: honey-gold, with wasp stripes wrapped all the way around a big round abdomen, and small
eyes. Side by side with a photo of a real *Drosophila melanogaster*, it was the wrong animal. It now has:

- **The eyes.** About three times the area, bright red, filling most of the head, the way a real fruit fly's do. This is
  the feature everyone recognises, and it was the most understated.
- **A fruit fly's colours:** a pale yellow-tan body with a faint grey cast, instead of honey gold.
- **A fruit fly's abdomen:** shorter, tapering to a dark tip, with dark bands only across the **top** of each segment,
  fading out on the underside and at the tip, instead of solid bands right around it.
- **Bristles** on the thorax and head, thinner and paler legs, clearer and longer wings, and the small scutellum behind
  the wing bases.

Only the drawing changed. The simulation, the physics, what each tool does and where a hit lands are all untouched, so
every number in the README still holds.

## For researchers

Nothing in the model, the connectome or the Lab tools changed in this release: the brain-view fibers and skeletons, the
backends, determinism and validation results are all exactly as in 2.8.0. The change is in `kickthefly/game/kick3d.py`
(`_draw_fly`) and the abdomen and eye patterns in `kickthefly/game/render3d.py`. Every README screenshot and the demo
GIF were regenerated from this build with `tools/make_screenshots.py`, which grew a `portrait` scene for checking the
model close up after a change.
