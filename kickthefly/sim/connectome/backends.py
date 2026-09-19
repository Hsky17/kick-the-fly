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
from typing import Any
import numpy as np
import scipy.sparse as sp

log = logging.getLogger("kickthefly")

BACKEND_NAMES = ("auto", "cpu", "numba", "torch-cuda", "torch-rocm")


class SimBackend:
    """Interface for pluggable LIF connectome simulator execution."""

    name: str = "base"
    device_name: str = "cpu"

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

    @numba.njit(fastmath=True, parallel=False)
    def _jit_csc_propagate(indices, indptr, data, active_cols, n_out):
        out = np.zeros(n_out, dtype=np.float32)
        for c in active_cols:
            start = indptr[c]
            end = indptr[c + 1]
            for idx in range(start, end):
                row = indices[idx]
                out[row] += data[idx]
        return out

    @numba.njit(fastmath=True)
    def _jit_step_core(v, refr, i_syn, noise_slice, sensory, gain, bias, ext_gain,
                       leak, v_reset, v_thresh, refractory_steps):
        n = len(v)
        spikes = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            drive = i_syn[i] * gain + bias + noise_slice[i]
            if sensory is not None:
                drive += sensory[i] * ext_gain
            v_val = v[i] * (1.0 - leak)
            if v_reset != 0.0:
                v_val += leak * v_reset
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
    """JIT-accelerated CPU backend for systems without GPUs."""

    name = "numba"
    device_name = "CPU (Numba JIT)"

    def setup(self) -> None:
        if not _numba_available:
            raise RuntimeError("Numba is not installed")

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        active = np.flatnonzero(sim.spikes)
        k_active = len(active)
        if k_active == 0:
            i_syn = sim._zeros
        elif k_active <= sim.p.sparse_path_max_active * self.n:
            sim.path_counts["columns"] += 1
            csc = sim.W_csc
            i_syn = _jit_csc_propagate(csc.indices, csc.indptr, csc.data, active, self.n)
        else:
            sim.path_counts["full"] += 1
            sim._sfloat[:] = sim.spikes
            i_syn = sim.W_csr @ sim._sfloat

        off = int(sim.rng.integers(0, sim._noise.size - self.n))
        noise_slice = sim._noise[off:off + self.n]

        sensory = None
        if sensory_input is not None:
            sensory = sensory_input if sensory_input.dtype == np.float32 else sensory_input.astype(np.float32)

        spikes = _jit_step_core(
            sim.v, sim.refr, i_syn, noise_slice, sensory,
            float(sim.gain), float(p.bias), float(p.ext_gain if p.ext_gain != 1.0 else 1.0),
            float(sim.leak), float(p.v_reset), float(p.v_thresh), int(p.refractory_steps)
        )
        sim.spikes = spikes
        return spikes


# --- PyTorch Backend (CUDA, ROCm, CPU) ------------------------------------------
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    torch = None


class TorchBackend(SimBackend):
    """PyTorch sparse simulation backend resident on device (CUDA / ROCm / CPU)."""

    name = "torch"

    def __init__(self, sim: Any, device: str = "cpu") -> None:
        super().__init__(sim)
        self.device_str = device
        self.device = None
        self.W_torch = None
        self.v_torch = None
        self.refr_torch = None
        self.spikes_torch = None

    def setup(self) -> None:
        if not _torch_available:
            raise RuntimeError("PyTorch is not installed")
        self.device = torch.device(self.device_str)
        dev_type = self.device.type
        if dev_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is False")
            prop = torch.cuda.get_device_properties(self.device)
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            self.device_name = f"{'ROCm' if is_rocm else 'CUDA'}: {prop.name}"
        else:
            self.device_name = f"PyTorch ({dev_type})"

        # Convert W_csr to torch sparse CSR tensor on device
        self._upload_weights()
        self.v_torch = torch.from_numpy(self.sim.v.astype(np.float32)).to(self.device)
        self.refr_torch = torch.from_numpy(self.sim.refr).to(self.device)
        self.spikes_torch = torch.zeros(self.n, dtype=torch.bool, device=self.device)

    def _upload_weights(self) -> None:
        csr = self.sim.W_csr
        crow_indices = torch.from_numpy(csr.indptr).to(torch.int64)
        col_indices = torch.from_numpy(csr.indices).to(torch.int64)
        values = torch.from_numpy(csr.data.astype(np.float32))
        self.W_torch = torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size=(self.n, self.n), device=self.device
        )

    def on_weights_changed(self) -> None:
        if self.W_torch is not None:
            self._upload_weights()

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        # 1. Sparse propagation on device: W_torch @ spikes_float
        s_float = self.spikes_torch.to(dtype=torch.float32).unsqueeze(1)
        i_syn = torch.sparse.mm(self.W_torch, s_float).squeeze(1)

        # 2. Add drive, noise, bias
        off = int(sim.rng.integers(0, sim._noise.size - self.n))
        noise_np = sim._noise[off:off + self.n]
        noise_t = torch.from_numpy(noise_np.astype(np.float32)).to(self.device)

        drive = i_syn * float(sim.gain) + float(p.bias) + noise_t
        if sensory_input is not None:
            sens_t = torch.from_numpy(sensory_input.astype(np.float32)).to(self.device)
            if p.ext_gain != 1.0:
                drive += float(p.ext_gain) * sens_t
            else:
                drive += sens_t

        # 3. Voltage state update
        v = self.v_torch
        refr = self.refr_torch
        v = v * (1.0 - float(sim.leak))
        if p.v_reset:
            v += float(sim.leak) * float(p.v_reset)
        v += drive

        # Apply refractory mask
        refr_mask = refr > 0
        v = torch.where(refr_mask, torch.tensor(float(p.v_reset), device=self.device), v)
        refr = torch.where(refr_mask, refr - 1, refr)

        # Spiking & reset
        new_spikes = v >= float(p.v_thresh)
        v = torch.where(new_spikes, torch.tensor(float(p.v_reset), device=self.device), v)
        refr = torch.where(new_spikes, torch.tensor(int(p.refractory_steps), dtype=torch.int16, device=self.device), refr)

        self.v_torch = v
        self.refr_torch = refr
        self.spikes_torch = new_spikes

        # Transfer only spikes to host
        spikes_np = new_spikes.cpu().numpy()
        sim.spikes = spikes_np
        # Sync back numpy arrays for inspector / savestates
        sim.v[:] = v.cpu().numpy()
        sim.refr[:] = refr.cpu().numpy()
        return spikes_np


def detect_available_backends() -> dict[str, str]:
    """Probes system and returns available backends mapping to human-readable device names."""
    avail = {"cpu": "CPU (NumPy)"}
    if _numba_available:
        avail["numba"] = "CPU (Numba JIT)"
    if _torch_available:
        avail["torch-cpu"] = "PyTorch (CPU)"
        if torch.cuda.is_available():
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            try:
                name = torch.cuda.get_device_name(0)
                if is_rocm:
                    avail["torch-rocm"] = f"ROCm: {name}"
                else:
                    avail["torch-cuda"] = f"CUDA: {name}"
            except Exception:
                if is_rocm:
                    avail["torch-rocm"] = "ROCm (AMD GPU)"
                else:
                    avail["torch-cuda"] = "CUDA (NVIDIA GPU)"
    return avail


def create_backend(sim: Any, backend_choice: str = "auto") -> SimBackend:
    """Creates requested backend with graceful fallback to CPU if unavailable."""
    choice = (backend_choice or "auto").lower()

    if choice == "auto":
        # Prefer GPU if available, then Numba, then CPU
        if _torch_available and torch.cuda.is_available():
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            target = "torch-rocm" if is_rocm else "torch-cuda"
            try:
                b = TorchBackend(sim, device="cuda:0")
                b.setup()
                log.info("Backend auto-detected: %s (%s)", target, b.device_name)
                return b
            except Exception as e:
                log.warning("Torch GPU backend failed to initialize (%s); falling back", e)
        if _numba_available:
            try:
                b = NumbaBackend(sim)
                b.setup()
                log.info("Backend auto-detected: numba (%s)", b.device_name)
                return b
            except Exception as e:
                log.warning("Numba backend failed (%s); falling back to CPU", e)
        return CPUBackend(sim)

    if choice in ("torch-cuda", "torch-rocm"):
        if not _torch_available:
            log.warning("PyTorch is not installed; falling back to CPU")
            return CPUBackend(sim)
        if not torch.cuda.is_available():
            log.warning("GPU acceleration requested (%s) but torch.cuda is unavailable; falling back to CPU", choice)
            return CPUBackend(sim)
        try:
            b = TorchBackend(sim, device="cuda:0")
            b.setup()
            return b
        except Exception as e:
            log.warning("Failed to start %s backend (%s); falling back to CPU", choice, e)
            return CPUBackend(sim)

    if choice == "numba":
        if not _numba_available:
            log.warning("Numba is not installed; falling back to CPU")
            return CPUBackend(sim)
        try:
            b = NumbaBackend(sim)
            b.setup()
            return b
        except Exception as e:
            log.warning("Failed to start Numba backend (%s); falling back to CPU", e)
            return CPUBackend(sim)

    return CPUBackend(sim)
