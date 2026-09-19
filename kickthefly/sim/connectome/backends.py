"""Pluggable simulation backends for Kick the Fly connectome simulation.

Provides backend interface and implementations:
- CPUBackend: Reference NumPy CPU implementation using CSC column gathering / CSR matvec
- NumbaBackend: JIT-compiled hot loops for CPU execution
- TorchBackend: PyTorch device-resident sparse ops for CUDA / ROCm / CPU
- Backend discovery, selection, and probing with automatic CPU fallback
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Any
import numpy as np
import scipy.sparse as sp

log = logging.getLogger("kickthefly")

BACKEND_NAMES = ("auto", "cpu", "numba", "torch-cpu", "torch-cuda", "torch-rocm")


class SimBackend:
    """Interface for pluggable LIF connectome simulator execution."""

    name: str = "base"
    device_name: str = "cpu"

    @property
    def device(self) -> str:
        """Human-readable device, for the Lab header, benchmarks, save states, crash reports and NWB metadata."""
        return self.device_name

    def __init__(self, sim: Any) -> None:
        self.sim = sim
        self.n = sim.n

    def setup(self) -> None:
        """Initialize backend-specific data structures or device allocations."""
        pass

    def on_weights_changed(self) -> None:
        """Notify backend that W_csr/W_csc matrix weights have been modified."""
        pass

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        """Advance one LIF step. Returns boolean spike vector of length n."""
        raise NotImplementedError


class CPUBackend(SimBackend):
    """Reference CPU / NumPy implementation (preserves exact single-vector math)."""

    name = "cpu"
    device_name = "CPU (NumPy)"

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        i_syn = sim._propagate()
        drive = sim._drive
        np.multiply(i_syn, dt(sim.gain), out=drive)
        drive += dt(p.bias)
        off = int(sim.rng.integers(0, sim._noise.size - self.n))
        drive += sim._noise[off:off + self.n]
        if sensory_input is not None:
            if p.ext_gain == 1.0:
                drive += sensory_input
            else:
                drive += dt(p.ext_gain) * sensory_input

        v = sim.v
        v *= dt(1.0 - sim.leak)
        if p.v_reset:
            v += sim.leak * dt(p.v_reset)
        v += drive
        mask = sim._mask
        np.greater(sim.refr, 0, out=mask)
        np.copyto(v, dt(p.v_reset), where=mask)
        np.subtract(sim.refr, 1, out=sim.refr, where=mask)
        spikes = v >= p.v_thresh
        np.copyto(v, dt(p.v_reset), where=spikes)
        np.copyto(sim.refr, np.int16(p.refractory_steps), where=spikes)
        sim.spikes = spikes
        return spikes


# --- Numba Backend -------------------------------------------------------------
_numba_available = False
_jit_step_core = None
_jit_csc_propagate = None

try:
    import numba

    # nogil: each fly's brain thread runs its kernels in parallel with the others. No fastmath, and every scalar arrives
    # already cast to the state dtype: the kernels must round exactly like the NumPy reference (same operations, same
    # order, same precision), or a chaotic network drifts apart in ~300 steps.
    @numba.njit(cache=True, nogil=True)
    def _jit_csc_propagate(indices, indptr, data, active_cols, n_out):
        out = np.zeros(n_out, dtype=data.dtype)
        for c in active_cols:                     # scipy's CSC matvec order: column by column, rows within a column
            for idx in range(indptr[c], indptr[c + 1]):
                out[indices[idx]] += data[idx]
        return out

    @numba.njit(cache=True, nogil=True)
    def _jit_step_core(v, refr, i_syn, noise_slice, sensory, has_sensory, gain, bias, ext_gain, scale_sensory,
                       keep, leak_reset, has_reset, v_reset, v_thresh, refractory_steps):
        n = len(v)
        spikes = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            drive = i_syn[i] * gain
            drive += bias
            drive += noise_slice[i]
            if has_sensory:
                if scale_sensory:
                    drive += ext_gain * sensory[i]
                else:
                    drive += sensory[i]
            v_val = v[i] * keep
            if has_reset:
                v_val += leak_reset
            v_val += drive
            if refr[i] > 0:
                v_val = v_reset
                refr[i] -= 1
            if v_val >= v_thresh:
                spikes[i] = True
                v[i] = v_reset
                refr[i] = refractory_steps
            else:
                v[i] = v_val
        return spikes

    _numba_available = True
except ImportError:
    pass


class NumbaBackend(SimBackend):
    """JIT-compiled CPU backend. Bit-exact with CPUBackend: same float operations in the same order."""

    name = "numba"
    device_name = "CPU (Numba JIT)"

    def setup(self) -> None:
        if not _numba_available:
            raise RuntimeError("Numba is not installed")

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        active = np.flatnonzero(sim.spikes)
        k_active = len(active)
        if k_active == 0:
            i_syn = sim._zeros
        elif k_active <= p.sparse_path_max_active * self.n:
            sim.path_counts["columns"] += 1
            csc = sim.W_csc
            i_syn = _jit_csc_propagate(csc.indices, csc.indptr, csc.data, active, self.n)
        else:
            sim.path_counts["full"] += 1
            sim._sfloat[:] = sim.spikes
            i_syn = sim.W_csr @ sim._sfloat

        off = int(sim.rng.integers(0, sim._noise.size - self.n))
        noise_slice = sim._noise[off:off + self.n]
        has_sensory = sensory_input is not None
        sensory = sensory_input if has_sensory else sim._zeros
        spikes = _jit_step_core(
            sim.v, sim.refr, i_syn, noise_slice, sensory, has_sensory,
            dt(sim.gain), dt(p.bias), dt(p.ext_gain), p.ext_gain != 1.0,
            dt(1.0 - sim.leak), sim.leak * dt(p.v_reset), bool(p.v_reset), dt(p.v_reset), dt(p.v_thresh),
            np.int16(p.refractory_steps))
        sim.spikes = spikes
        return spikes


# --- PyTorch Backend (CUDA, ROCm, CPU) ------------------------------------------
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    torch = None


def _torch_gpu_kind() -> str | None:
    """'torch-rocm' or 'torch-cuda' for the GPU this torch build can use, or None."""
    if not _torch_available or not torch.cuda.is_available():
        return None
    return "torch-rocm" if getattr(torch.version, "hip", None) else "torch-cuda"


class TorchBackend(SimBackend):
    """PyTorch backend: the weight matrix lives on the device (CUDA, ROCm or CPU) and the sparse product and state
    update run there. The host arrays (sim.v, sim.refr, sim.spikes) stay the source of truth: they are uploaded at the
    start of each step and written back at the end, so save states, the neural clamp and the inspector, which read and
    write them directly, work unchanged. That costs ~1 MB of transfers per step."""

    name = "torch"

    def __init__(self, sim: Any, device: str = "cpu") -> None:
        super().__init__(sim)
        self.device_str = device
        self.tdev = None
        self.W_torch = None

    def setup(self) -> None:
        if not _torch_available:
            raise RuntimeError("PyTorch is not installed")
        self.tdev = torch.device(self.device_str)
        if self.tdev.type == "cuda":
            kind = _torch_gpu_kind()
            if kind is None:
                raise RuntimeError("a GPU was requested but torch.cuda.is_available() is False")
            self.name = kind
            self.device_name = f"{'ROCm' if kind == 'torch-rocm' else 'CUDA'}: {torch.cuda.get_device_name(self.tdev)}"
        else:
            self.name = f"torch-{self.tdev.type}"
            self.device_name = f"PyTorch ({self.tdev.type})"
        self._upload_weights()

    def _upload_weights(self) -> None:
        csr = self.sim.W_csr
        with warnings.catch_warnings():             # "sparse CSR support is in beta", on every learning step otherwise
            warnings.simplefilter("ignore", UserWarning)
            self.W_torch = self._csr_tensor(csr)
        self._W64 = None                            # rebuilt from W_torch on the next busy float64 step

    def _csr_tensor(self, csr):
        return torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(np.int64)), torch.from_numpy(csr.indices.astype(np.int64)),
            torch.from_numpy(np.ascontiguousarray(csr.data, dtype=np.float32)), size=(self.n, self.n),
            device=self.tdev, check_invariants=False)

    def on_weights_changed(self) -> None:
        if self.W_torch is not None:
            self._upload_weights()

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        tdt = torch.float64 if dt is np.float64 else torch.float32
        dev = self.tdev
        if self.W_torch is None or self.W_torch.values().numel() != sim.W_csr.nnz:
            self._upload_weights()                   # the matrix object was replaced (mirror weights, a new wiring)

        # The same operations in the same order and precision as CPUBackend. A CSR row product adds a row's inputs in
        # column order, which is the order the CPU's column gather adds them in, so on the CPU device the result is
        # bit-exact. GPU sparse kernels may sum in a different order: see docs on determinism across backends.
        # The reference sums a sparse step's inputs in float32 (its column path) but a busy step's (more than
        # sparse_path_max_active firing) in the state dtype (its dense path); follow it, or float64 runs drift.
        dense64 = tdt == torch.float64 and np.count_nonzero(sim.spikes) > p.sparse_path_max_active * self.n
        if dense64 and getattr(self, "_W64", None) is None:
            self._W64 = self.W_torch.to(torch.float64)
        W = self._W64 if dense64 else self.W_torch
        s_float = torch.from_numpy(sim.spikes).to(dev).to(W.dtype).unsqueeze(1)
        i_syn = torch.sparse.mm(W, s_float).squeeze(1)
        drive = i_syn.to(tdt) * torch.tensor(dt(sim.gain), device=dev)
        drive += torch.tensor(dt(p.bias), device=dev)
        off = int(sim.rng.integers(0, sim._noise.size - self.n))
        drive += torch.from_numpy(sim._noise[off:off + self.n]).to(dev)
        if sensory_input is not None:
            sens = torch.from_numpy(np.ascontiguousarray(sensory_input)).to(dev)
            if p.ext_gain == 1.0:
                drive += sens.to(tdt)
            else:
                drive += (torch.tensor(dt(p.ext_gain), device=dev) * sens).to(tdt)

        v = torch.from_numpy(sim.v).to(dev)
        refr = torch.from_numpy(sim.refr).to(dev)
        v = v * torch.tensor(dt(1.0 - sim.leak), device=dev)
        if p.v_reset:
            v += torch.tensor(sim.leak * dt(p.v_reset), device=dev)
        v += drive
        v_reset = torch.tensor(dt(p.v_reset), device=dev)
        refr_mask = refr > 0
        v = torch.where(refr_mask, v_reset, v)
        refr = torch.where(refr_mask, refr - 1, refr)
        spikes = v >= torch.tensor(dt(p.v_thresh), device=dev)
        v = torch.where(spikes, v_reset, v)
        refr = torch.where(spikes, torch.tensor(int(p.refractory_steps), dtype=refr.dtype, device=dev), refr)

        sim.v[:] = v.cpu().numpy()
        sim.refr[:] = refr.cpu().numpy()
        spikes_np = spikes.cpu().numpy()
        sim.spikes = spikes_np
        return spikes_np


def detect_available_backends() -> dict[str, str]:
    """The backends that can run here, mapped to human-readable device names."""
    avail = {"cpu": CPUBackend.device_name}
    if _numba_available:
        avail["numba"] = NumbaBackend.device_name
    if _torch_available:
        avail["torch-cpu"] = "PyTorch (cpu)"
        kind = _torch_gpu_kind()
        if kind:
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "AMD GPU" if kind == "torch-rocm" else "NVIDIA GPU"
            avail[kind] = f"{'ROCm' if kind == 'torch-rocm' else 'CUDA'}: {name}"
    return avail


def _try(make, label: str) -> SimBackend | None:
    try:
        b = make()
        b.setup()
        return b
    except Exception as e:
        log.warning("%s backend unavailable (%s)", label, e)
        return None


def create_backend(sim: Any, backend_choice: str = "auto") -> SimBackend:
    """The requested backend, or the best one available for 'auto', falling back to the NumPy CPU backend.
    The returned backend's .name is what actually runs (e.g. 'torch-rocm' when 'torch-cuda' was asked for on a ROCm
    build of PyTorch, or 'cpu' after a fallback), and that is what gets recorded everywhere."""
    choice = (backend_choice or "auto").lower()
    b: SimBackend | None = None
    if choice == "auto":
        if _torch_gpu_kind():
            b = _try(lambda: TorchBackend(sim, "cuda:0"), "PyTorch GPU")
        if b is None and _numba_available:
            b = _try(lambda: NumbaBackend(sim), "Numba")
    elif choice in ("torch-cuda", "torch-rocm"):
        kind = _torch_gpu_kind()
        if not _torch_available:
            log.warning("%s requested but PyTorch is not installed; using the CPU backend", choice)
        elif kind is None:
            log.warning("%s requested but this PyTorch build sees no GPU (torch %s); using the CPU backend",
                        choice, torch.__version__)
        else:
            if kind != choice:
                log.warning("%s requested; this PyTorch build is %s, using that", choice, kind)
            b = _try(lambda: TorchBackend(sim, "cuda:0"), choice)
    elif choice == "torch-cpu":
        if not _torch_available:
            log.warning("torch-cpu requested but PyTorch is not installed; using the CPU backend")
        else:
            b = _try(lambda: TorchBackend(sim, "cpu"), "PyTorch CPU")
    elif choice == "numba":
        if not _numba_available:
            log.warning("numba requested but Numba is not installed; using the CPU backend")
        else:
            b = _try(lambda: NumbaBackend(sim), "Numba")
    elif choice != "cpu":
        log.warning("unknown simulation backend %r; using the CPU backend", choice)
    if b is None:
        b = CPUBackend(sim)
    try:
        from kickthefly.core import crash
        crash.record_backend(b.name, b.device)
    except Exception:
        pass
    return b
