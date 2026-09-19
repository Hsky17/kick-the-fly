"""Simulation performance benchmark for 1, 8, and 16 flies.

Measures paced steps/s (real-time keeping), uncapped throughput, neurons/second,
synapse updates/second, sim-time vs real-time ratio, memory footprint, and CPU/system info.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
import numpy as np

from kickthefly.sim import brainpack
from kickthefly.core import config
from kickthefly.sim.connectome.sim import LIFParams, LIFSim
from kickthefly.core import paths


BENCHMARK_FILE = "benchmark_results.json"


def get_system_info() -> dict:
    info = {
        "os": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "python": platform.python_version(),
    }
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    if "cpu_model" not in info:
        info["cpu_model"] = platform.processor() or "Unknown CPU"
    return info


def get_memory_mb() -> float:
    """The process's current resident memory (Linux), else its peak (other systems)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except (OSError, ValueError, AttributeError):
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux, ru_maxrss is in kilobytes
        if platform.system() == "Darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except Exception:
        return 0.0


def make_bench_brain(g, W, seed: int, backend: str = "auto"):
    from kickthefly.game import kick_the_fly as k
    p = LIFParams()
    p.backend = backend
    sim = LIFSim(None, p, W_in=W, seed=seed)
    br = k.Brain(g, sim, seed=seed)
    if getattr(g, "dan_mbon", None) is not None:
        from kickthefly.core import memory
        os.environ.setdefault("KICK_THE_FLY_MEMORY", str(Path(os.environ.get("TMPDIR", "/tmp")) / "ktf-bench-mem"))
        br.memory = memory.Memory(g, sim)
        br.memory.save = lambda: None
    br.warmup(50)
    return br


def measure_steps(brains, seconds: float, speed: float):
    """Steps/s per brain, and the spikes they fired, over `seconds` of wall-clock time."""
    for b in brains:
        b.speed = speed
    s0 = [b.steps for b in brains]
    k0 = [b.sim.spike_total for b in brains]
    t0 = time.perf_counter()
    for b in brains:
        b.start()
    time.sleep(seconds)
    dt = time.perf_counter() - t0
    for b in brains:
        b.stop()
    time.sleep(0.15)
    rates = [(b.steps - s) / dt for b, s in zip(brains, s0)]
    spikes = sum(b.sim.spike_total - k for b, k in zip(brains, k0))
    steps = sum(b.steps - s for b, s in zip(brains, s0))
    return rates, spikes / max(1, steps)


def run_benchmark(fly_counts: tuple[int, ...] = (1, 8, 16), seconds: float = 3.0, progress_cb=None,
                  backend: str = "auto") -> dict:
    g, W, _ = brainpack.load(brainpack.find())
    sys_info = get_system_info()
    n_neurons = int(g.n)
    n_synapses = int(W.nnz)

    records = []
    total_runs = len(fly_counts)
    active_backend = "Unknown"
    active_device = "Unknown"

    for idx, n in enumerate(fly_counts):
        if progress_cb:
            progress_cb(idx, total_runs, f"benchmarking {n} flies")

        brains = [make_bench_brain(g, W, seed=s, backend=backend) for s in range(n)]
        if idx == 0 and brains:
            b_obj = getattr(brains[0].sim, "backend", None)
            if b_obj is not None:
                active_backend = b_obj.name                      # what actually ran, after any fallback
                active_device = b_obj.device

        # 1. Paced at real-time (speed = 1.0)
        paced_rates, _ = measure_steps(brains, seconds, 1.0)
        mean_paced = float(np.mean(paced_rates))
        min_paced = float(np.min(paced_rates))
        rt_target = 1000.0 / brains[0].sim.p.dt_ms  # 200.0 steps/s
        paced_ratio = mean_paced / rt_target

        # 2. Uncapped (speed = 1000.0)
        for b in brains:
            b._stop = False
        uncapped_rates, spikes_per_step = measure_steps(brains, seconds, 1000.0)
        mean_uncapped = float(np.mean(uncapped_rates))
        agg_uncapped_steps = float(np.sum(uncapped_rates))
        uncapped_ratio = mean_uncapped / rt_target

        # Neuron updates: every neuron is integrated every step. Synaptic events: each spike is delivered to all of its
        # outgoing synapses; the spikes are counted during the run, the out-degree is the connectome's mean (so this
        # is spikes/s x 61.6, not a count of multiply-adds, which depends on the path the step took).
        neurons_per_sec = agg_uncapped_steps * n_neurons
        synapses_per_sec = agg_uncapped_steps * spikes_per_step * (n_synapses / max(1, n_neurons))

        mem_mb = get_memory_mb()

        records.append({
            "flies": n,
            "paced_steps_per_s": round(mean_paced, 1),
            "paced_min_steps_per_s": round(min_paced, 1),
            "paced_realtime_ratio": round(paced_ratio, 3),
            "uncapped_steps_per_s": round(mean_uncapped, 1),
            "uncapped_realtime_ratio": round(uncapped_ratio, 2),
            "agg_uncapped_steps_per_s": round(agg_uncapped_steps, 1),
            "neurons_per_sec": round(neurons_per_sec, 0),
            "active_fraction": round(spikes_per_step / n_neurons, 4),
            "synapses_per_sec": round(synapses_per_sec, 0),
            "memory_mb": round(mem_mb, 1),
        })
        del brains

    if progress_cb:
        progress_cb(total_runs, total_runs, "complete")

    res = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": sys_info,
        "backend": active_backend,
        "device": active_device,
        "connectome": {
            "neurons": n_neurons,
            "synapses": n_synapses,
        },
        "seconds_per_run": seconds,
        "records": records,
    }
    return res


def format_benchmark_report(res: dict) -> str:
    sys_info = res["system"]
    con = res["connectome"]
    backend = res.get("backend", "Unknown")
    device = res.get("device", "Unknown")
    lines = [
        "=" * 106,
        "KICK THE FLY - SIMULATION PERFORMANCE BENCHMARK",
        f"Timestamp: {res['timestamp']}  |  CPU: {sys_info.get('cpu_model')} ({sys_info.get('cpu_count')} threads)",
        f"OS: {sys_info.get('os')}  |  Python: {sys_info.get('python')}",
        f"Backend: {backend}  |  Device: {device}",
        f"Connectome: MaleCNS v1.0 ({con['neurons']:,} neurons, {con['synapses']:,} synapses)",
        "=" * 106,
        f"{'Flies':<6} {'Paced (steps/s)':<17} {'Sim/Real':<10} {'Uncapped':<16} {'Speedup':<9} {'Neurons/s':<16} {'Syn-events/s':<16} {'Memory':<8}",
        "-" * 106,
    ]
    for r in res["records"]:
        fl = str(r["flies"])
        paced = f"{r['paced_steps_per_s']:.1f} (min {r['paced_min_steps_per_s']:.1f})"
        rt = f"{r['paced_realtime_ratio']:.2f}x"
        uncap = f"{r['uncapped_steps_per_s']:.1f}/fly"
        speedup = f"{r['uncapped_realtime_ratio']:.2f}x"
        n_sec = f"{r['neurons_per_sec'] / 1e6:.1f} M/s"
        syn_sec = f"{r.get('synapses_per_sec', 0) / 1e6:.1f} M/s"
        mem = f"{r['memory_mb']:.0f} MB"
        lines.append(f"{fl:<6} {paced:<17} {rt:<10} {uncap:<16} {speedup:<9} {n_sec:<16} {syn_sec:<16} {mem:<8}")
    lines.append("=" * 106)
    lines.append("Note: Real-time pace requires 200 steps/s (1.00x). Values >= 1.00x run in true real-time.")
    lines.append("Neurons/s: neuron updates per wall-clock second, all flies together. Syn-events/s: measured spikes")
    lines.append("per second x the connectome's mean out-degree. Memory: the process's resident memory after the run.")
    return "\n".join(lines)


def default_results_path() -> Path:
    p = paths.get()
    return p.state_dir / BENCHMARK_FILE


def save_benchmark_results(res: dict, path: Path | str | None = None) -> Path:
    p = Path(path) if path else default_results_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    return p


def load_benchmark_results(path: Path | str | None = None) -> dict | None:
    p = Path(path) if path else default_results_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
