# Kick the Fly 2.8.3: GPU backends and bigger swarms

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over. Everything in
[2.8.2](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/tag/v2.8.2) is in this build.

## For players

**More flies at once.** On a machine with a GPU the swarm cap now scales up to 32-64 flies instead of the old fixed
limit. Every one of them is running the same full connectome simulation as before; there are just more of them.

**The targeted laser responds immediately.** Holding the laser on a fly used to cost a noticeable hitch every frame.
That hitch is gone. The laser does exactly what it did before, only without the stall.

**Nothing you do in the game changed.** No new controls, no changed behaviour, no altered results. This release is
about how fast the simulation runs, not what it does.

## For researchers

The simulation gained three GPU execution paths, selectable with `--backend`:

- **`gl`**: a vendor-neutral OpenGL 4.3+ compute shader backend (`cs_spmv`, `cs_lif`) using SSBOs, with explicit
  memory barriers between the sparse accumulation and membrane-update passes. It needs no PyTorch at all.
- **`torch-rocm` / `torch-cuda`**: membrane potentials, refractory counters, spike buffers and pre-scaled noise stay
  resident in VRAM across steps, and the connectome sparse matrix is streamed once per step across N flies rather
  than once per fly.
- Optional `torch.compile` elementwise fusion for the LIF update, behind a toggle.

Backend selection falls back cleanly: Torch GPU to GL to Numba to NumPy CPU, covering OpenGL older than 4.3, missing
ModernGL, absent PyTorch and failed GL context creation. GL contexts are allocated thread-locally so parallel brain
workers do not serialise on a shared context.

`--benchmark` reports, the JSON exports and the Lab benchmark dashboard now carry per-fly uncapped step latency in ms.

`kickthefly/lab/laser.py:resolve_target_rows` was rescanning the full 166,700-entry cell-type string table on every
`apply()` call, once per fly per frame. The matching rows depend only on `brain.types`, which is fixed for a brain's
lifetime, so the lookup is now cached per brain and keyed by target type. The matching itself is unchanged: the same
exact, then prefix, then substring precedence, returning the same rows. Found and fixed by
[@Hsky17](https://github.com/Hsky17) in [#1](https://github.com/legendarylolo318-cloud/kick-the-fly/pull/1).

Simulation results are unchanged. The backends are validated bit-exact against the CPU path in `tests/test_backends.py`.
