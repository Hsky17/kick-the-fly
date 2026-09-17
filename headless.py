"""Headless runs: no window, no sound, no display needed (SSH, CI, a Windows console).

    KickTheFly --headless --validate [--out results.json] [--workers N] [--seeds 1000-1009]
    KickTheFly --headless --protocol experiment.yaml [--out folder]

The exe and the AppImage accept the same flags. On Windows the exe borrows the console it was started from for its
output; from cmd use `start /wait KickTheFly.exe --headless ...` (or check the files written to --out).
Exit codes: 0 success; 1 a validation result differs from the expected pass/fail (only with --strict); 2 bad input.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from crash import log


def prepare() -> None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"          # nothing in a headless run may open a window or an audio device
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import platform_env

    platform_env.attach_console()


def parse_seeds(text: str | None, default):
    if not text:
        return tuple(default)
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return tuple(out)


def run_validate(args) -> int:
    import validation

    seeds = parse_seeds(args.seeds, validation.SEEDS)
    t0 = time.time()

    def progress(done, total, label):
        print(f"  {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    res = validation.run(seeds=seeds, workers=args.workers, progress=progress)
    out = Path(args.out) if args.out else validation.local_path()
    if out.suffix.lower() != ".json":
        out = out / validation.RESULTS_NAME
    validation.save_results(res, out)
    print(validation.summary(res))
    print(f"results written to {out}")
    if args.strict:
        wrong = [t["id"] for t in res["tests"] if t["passed"] != validation.EXPECTED.get(t["id"])]
        if wrong:
            print(f"differs from the expected results: {', '.join(wrong)}")
            return 1
    return 0


def audit_asymmetry(seconds: float = 5.0, seed: int = 0, mirror: bool = False) -> dict:
    """Audit bilateral asymmetry between left and right hemibrains:
    - Measures baseline turning bias with no input over a calm run
    - Compares L vs R synapse in/out counts and firing rates for key cell types:
      DNa01, DNa02, LC10, LPLC2, LC4, DNp01.
    """
    import numpy as np
    import simcore
    from connectome.sim import LIFParams, LIFSim

    g, W, _ = simcore.pack()
    if mirror:
        W = simcore.symmetrize_weights(g, W)
    inst = g.instance.astype(str)
    types = g.type.astype(str)
    W_csr = W.tocsr()
    W_csc = W.tocsc()

    sim = LIFSim(None, LIFParams(), W_in=W, seed=seed)
    for _ in range(100):
        sim.step()

    steps = int(round(seconds / 0.005))
    spikes = np.zeros(g.n, dtype=np.int32)
    for _ in range(steps):
        sim.step()
        spikes += sim.spikes
    rates = spikes / float(seconds)

    key_types = ["DNa01", "DNa02", "LC10", "LPLC2", "LC4", "DNp01"]
    records = []
    for t in key_types:
        tm = np.char.startswith(types, t)
        l_idx = np.flatnonzero(tm & np.char.endswith(inst, "_L"))
        r_idx = np.flatnonzero(tm & np.char.endswith(inst, "_R"))
        l_n, r_n = len(l_idx), len(r_idx)
        l_in = float(sum(W_csr[i].nnz for i in l_idx) / l_n) if l_n else 0.0
        r_in = float(sum(W_csr[i].nnz for i in r_idx) / r_n) if r_n else 0.0
        l_out = float(sum(W_csc[:, i].nnz for i in l_idx) / l_n) if l_n else 0.0
        r_out = float(sum(W_csc[:, i].nnz for i in r_idx) / r_n) if r_n else 0.0
        l_hz = float(rates[l_idx].mean()) if l_n else 0.0
        r_hz = float(rates[r_idx].mean()) if r_n else 0.0
        records.append({
            "type": t,
            "l_count": l_n, "r_count": r_n,
            "l_in_syn": round(l_in, 1), "r_in_syn": round(r_in, 1),
            "l_out_syn": round(l_out, 1), "r_out_syn": round(r_out, 1),
            "l_rate_hz": round(l_hz, 2), "r_rate_hz": round(r_hz, 2),
            "diff_rate_hz": round(r_hz - l_hz, 2),
        })

    dna_r = rates[np.flatnonzero(np.isin(types, ("DNa01", "DNa02")) & np.char.endswith(inst, "_R"))].mean()
    dna_l = rates[np.flatnonzero(np.isin(types, ("DNa01", "DNa02")) & np.char.endswith(inst, "_L"))].mean()
    turn_bias = float(dna_r - dna_l)

    return {
        "seconds": seconds,
        "seed": seed,
        "mirror": mirror,
        "turning_bias_hz": round(turn_bias, 3),
        "turning_direction": "right" if turn_bias > 0.05 else "left" if turn_bias < -0.05 else "neutral",
        "records": records,
    }


def format_asymmetry_report(res: dict) -> str:
    lines = [
        f"LEFT/RIGHT ASYMMETRY AUDIT ({res['seconds']:.1f} s calm run, seed={res['seed']}, mirror={res['mirror']})",
        "-" * 88,
        f"{'Cell Type':<9} {'Count L/R':<11} {'In-Syn L/R':<15} {'Out-Syn L/R':<15} {'Firing Rate L/R':<18} {'Diff (R-L)':<10}",
        "-" * 88,
    ]
    for r in res["records"]:
        counts = f"{r['l_count']}/{r['r_count']}"
        in_s = f"{r['l_in_syn']:.0f}/{r['r_in_syn']:.0f}"
        out_s = f"{r['l_out_syn']:.0f}/{r['r_out_syn']:.0f}"
        rates_s = f"{r['l_rate_hz']:.1f}/{r['r_rate_hz']:.1f} Hz"
        diff_s = f"{r['diff_rate_hz']:+.2f} Hz"
        lines.append(f"{r['type']:<9} {counts:<11} {in_s:<15} {out_s:<15} {rates_s:<18} {diff_s:<10}")
    lines.append("-" * 88)
    bias = res["turning_bias_hz"]
    direction = res["turning_direction"]
    lines.append(f"Baseline Steering Bias (DNa01/02 R - L): {bias:+.2f} Hz ({direction} turn bias)")
    if res["mirror"]:
        lines.append("Note: Mirror-averaged weights active [GAME RULE: data modification].")
    else:
        lines.append("Note: Raw connectome weights. Asymmetries stem from both biology and EM reconstruction depth.")
    return "\n".join(lines)


def run_audit_asymmetry(args) -> int:
    import json
    mirror = getattr(args, "mirror_weights", False)
    seed = getattr(args, "seed", None) or 0
    res = audit_asymmetry(seconds=5.0, seed=seed, mirror=mirror)
    print(format_asymmetry_report(res))
    if getattr(args, "out", None):
        out_path = Path(args.out)
        if out_path.is_dir() or not out_path.suffix:
            out_path = out_path / "asymmetry_audit.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2))
        print(f"audit results written to {out_path}")
    return 0


def main(args) -> int:
    prepare()
    log.info("headless run")
    try:
        if getattr(args, "audit_asymmetry", False):
            return run_audit_asymmetry(args)
        if args.validate:
            return run_validate(args)
        if args.protocol:
            import protocol

            return protocol.run_file(Path(args.protocol), Path(args.out) if args.out else None, workers=args.workers)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("nothing to do: use --validate or --protocol FILE or --audit-asymmetry", file=sys.stderr)
    return 2

