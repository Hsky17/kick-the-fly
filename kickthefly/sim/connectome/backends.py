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

BACKEND_NAMES = ("auto", "cpu", "numba", "torch-cpu", "torch-cuda", "torch-rocm", "gl")


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

    def sync_to_host(self) -> None:
        """Sync device-resident state (v, refr, spikes) back to host numpy arrays."""
        pass

    def sync_from_host(self) -> None:
        """Sync host numpy arrays (v, refr, spikes) up to device tensors."""
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


_compiled_lif_step = None


def _get_compiled_lif_step():
    global _compiled_lif_step
    if _compiled_lif_step is None and _torch_available and torch is not None:
        def _core(v, refr, i_syn, gain, bias, noise_slice, sens, ext_gain, leak_decay, leak_reset, v_reset, v_thresh, refr_steps):
            drive = i_syn * gain + bias + noise_slice
            if sens is not None:
                drive = drive + (sens if ext_gain == 1.0 else sens * ext_gain)
            v_new = v * leak_decay + leak_reset + drive
            refr_mask = refr > 0
            v_new = torch.where(refr_mask, v_reset, v_new)
            refr_new = torch.where(refr_mask, refr - 1, refr)
            spikes = v_new >= v_thresh
            v_out = torch.where(spikes, v_reset, v_new)
            refr_out = torch.where(spikes, refr_steps, refr_new)
            return v_out, refr_out, spikes

        try:
            _compiled_lif_step = torch.compile(_core)
        except Exception as e:
            log.warning("torch.compile unavailable for fused LIF: %s", e)
            _compiled_lif_step = _core
    return _compiled_lif_step



class TorchBackend(SimBackend):
    """PyTorch backend: state tensors (v, refr, spikes, noise bank) and weight matrix live on device
    (CUDA, ROCm, or CPU) across steps for zero-copy simulation. Host arrays (sim.v, sim.refr, sim.spikes)
    are synchronized on demand via sync_to_host() / sync_from_host()."""

    name = "torch"

    def __init__(self, sim: Any, device: str = "cpu") -> None:
        super().__init__(sim)
        self.device_str = device
        self.tdev = None
        self.tdt = None
        self.W_torch = None
        self._W64 = None
        self.v_dev = None
        self.refr_dev = None
        self.spikes_dev = None
        self.s_float_dev = None
        self.noise_dev = None
        self._noise_id = None
        # Hoisted parameter tensors
        self._bias_tensor = None
        self._leak_decay_tensor = None
        self._leak_reset_tensor = None
        self._v_reset_tensor = None
        self._v_thresh_tensor = None
        self._refr_steps_tensor = None
        self._ext_gain_tensor = None

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
        dt = self.sim.dtype
        self.tdt = torch.float64 if dt is np.float64 else torch.float32
        self._upload_weights()
        self._init_params()
        self.sync_from_host()

    def _init_params(self) -> None:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        dev = self.tdev
        tdt = self.tdt
        self._bias_tensor = torch.tensor(dt(p.bias), dtype=tdt, device=dev)
        self._leak_decay_tensor = torch.tensor(dt(1.0 - sim.leak), dtype=tdt, device=dev)
        self._leak_reset_tensor = torch.tensor(sim.leak * dt(p.v_reset), dtype=tdt, device=dev)
        self._v_reset_tensor = torch.tensor(dt(p.v_reset), dtype=tdt, device=dev)
        self._v_thresh_tensor = torch.tensor(dt(p.v_thresh), dtype=tdt, device=dev)
        self._refr_steps_tensor = torch.tensor(int(p.refractory_steps), dtype=torch.int16, device=dev)
        self._ext_gain_tensor = torch.tensor(dt(p.ext_gain), dtype=tdt, device=dev)

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

    def sync_to_host(self) -> None:
        if self.v_dev is not None:
            self.sim.v[:] = self.v_dev.cpu().numpy()
            self.sim.refr[:] = self.refr_dev.cpu().numpy()
            self.sim.spikes[:] = self.spikes_dev.cpu().numpy()

    def sync_from_host(self) -> None:
        if self.tdev is not None:
            sim = self.sim
            dev = self.tdev
            tdt = self.tdt or (torch.float64 if sim.dtype is np.float64 else torch.float32)
            self.v_dev = torch.from_numpy(sim.v).to(dev, dtype=tdt)
            self.refr_dev = torch.from_numpy(sim.refr).to(dev, dtype=torch.int16)
            self.spikes_dev = torch.from_numpy(sim.spikes).to(dev, dtype=torch.bool)
            self.s_float_dev = self.spikes_dev.to(self.W_torch.dtype if self.W_torch is not None else torch.float32).unsqueeze(1)
            self.noise_dev = torch.from_numpy(sim._noise).to(dev, dtype=tdt)
            self._noise_id = id(sim._noise)

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        p = sim.p
        dt = sim.dtype
        tdt = self.tdt
        dev = self.tdev
        if self.W_torch is None or self.W_torch.values().numel() != sim.W_csr.nnz:
            self._upload_weights()
        if self.v_dev is None:
            self.sync_from_host()
        if self._noise_id != id(sim._noise):
            self.noise_dev = torch.from_numpy(sim._noise).to(dev, dtype=tdt)
            self._noise_id = id(sim._noise)

        dense64 = (tdt == torch.float64) and (self.spikes_dev.sum().item() > p.sparse_path_max_active * self.n)
        if dense64 and getattr(self, "_W64", None) is None:
            self._W64 = self.W_torch.to(torch.float64)
        W = self._W64 if dense64 else self.W_torch

        s_float = self.spikes_dev.to(W.dtype).unsqueeze(1)
        i_syn = torch.sparse.mm(W, s_float).squeeze(1)
        off = int(sim.rng.integers(0, sim._noise.size - self.n))

        if getattr(p, "fuse_lif", False):
            fused_fn = _get_compiled_lif_step()
            sens = None
            if sensory_input is not None:
                sens = torch.from_numpy(np.ascontiguousarray(sensory_input)).to(dev, dtype=tdt)
            v, refr, spikes = fused_fn(
                self.v_dev, self.refr_dev, i_syn.to(tdt), sim.gain, self._bias_tensor,
                self.noise_dev[off:off + self.n], sens, p.ext_gain, self._leak_decay_tensor,
                self._leak_reset_tensor if p.v_reset else torch.zeros(1, dtype=tdt, device=dev),
                self._v_reset_tensor, self._v_thresh_tensor, self._refr_steps_tensor
            )
        else:
            drive = i_syn.to(tdt) * sim.gain
            drive += self._bias_tensor
            drive += self.noise_dev[off:off + self.n]

            if sensory_input is not None:
                sens = torch.from_numpy(np.ascontiguousarray(sensory_input)).to(dev, dtype=tdt)
                if p.ext_gain == 1.0:
                    drive += sens
                else:
                    drive += self._ext_gain_tensor * sens

            v = self.v_dev * self._leak_decay_tensor
            if p.v_reset:
                v += self._leak_reset_tensor
            v += drive

            refr_mask = self.refr_dev > 0
            v = torch.where(refr_mask, self._v_reset_tensor, v)
            refr = torch.where(refr_mask, self.refr_dev - 1, self.refr_dev)

            spikes = v >= self._v_thresh_tensor
            v = torch.where(spikes, self._v_reset_tensor, v)
            refr = torch.where(spikes, self._refr_steps_tensor, refr)

        self.v_dev = v
        self.refr_dev = refr
        self.spikes_dev = spikes

        spikes_np = spikes.cpu().numpy()
        sim.spikes = spikes_np
        return spikes_np

    @classmethod
    def step_batch(cls, sims: list[Any], sensory_inputs: list[np.ndarray | None] | None = None) -> list[np.ndarray]:
        """Advance a batch of LIF simulations on GPU with a single batched SpMM multiplication."""
        if not sims:
            return []
        if len(sims) == 1:
            sens = sensory_inputs[0] if sensory_inputs else None
            return [sims[0].step(sens)]

        backends = [s.backend for s in sims]
        for b in backends:
            if b.v_dev is None:
                b.sync_from_host()
            if b._noise_id != id(b.sim._noise):
                b.noise_dev = torch.from_numpy(b.sim._noise).to(b.tdev, dtype=b.tdt)
                b._noise_id = id(b.sim._noise)

        # Group simulations by weight tensor for batched SpMM
        groups: dict[Any, list[int]] = {}
        for idx, b in enumerate(backends):
            w = b.W_torch
            groups.setdefault(w, []).append(idx)

        i_syn_list = [None] * len(sims)
        for w, indices in groups.items():
            if len(indices) == 1:
                idx = indices[0]
                b = backends[idx]
                s_float = b.spikes_dev.to(w.dtype).unsqueeze(1)
                i_syn_list[idx] = torch.sparse.mm(w, s_float).squeeze(1)
            else:
                stacked_spikes = torch.stack([backends[idx].spikes_dev.to(w.dtype) for idx in indices], dim=1)
                batched_out = torch.sparse.mm(w, stacked_spikes)
                for col, idx in enumerate(indices):
                    i_syn_list[idx] = batched_out[:, col]

        results = []
        for idx, (sim, b) in enumerate(zip(sims, backends)):
            p = sim.p
            tdt = b.tdt
            dev = b.tdev
            i_syn = i_syn_list[idx]
            drive = i_syn.to(tdt) * sim.gain
            drive += b._bias_tensor
            off = int(sim.rng.integers(0, sim._noise.size - sim.n))
            drive += b.noise_dev[off:off + sim.n]

            sens = sensory_inputs[idx] if sensory_inputs else None
            if sens is not None:
                sens_t = torch.from_numpy(np.ascontiguousarray(sens)).to(dev, dtype=tdt)
                if p.ext_gain == 1.0:
                    drive += sens_t
                else:
                    drive += b._ext_gain_tensor * sens_t

            v = b.v_dev * b._leak_decay_tensor
            if p.v_reset:
                v += b._leak_reset_tensor
            v += drive

            refr_mask = b.refr_dev > 0
            v = torch.where(refr_mask, b._v_reset_tensor, v)
            refr = torch.where(refr_mask, b.refr_dev - 1, b.refr_dev)

            spikes = v >= b._v_thresh_tensor
            v = torch.where(spikes, b._v_reset_tensor, v)
            refr = torch.where(spikes, b._refr_steps_tensor, refr)

            b.v_dev = v
            b.refr_dev = refr
            b.spikes_dev = spikes

            spikes_np = spikes.cpu().numpy()
            sim.spikes = spikes_np

            fired = int(np.count_nonzero(spikes_np))
            sim.spike_total += fired
            frac = fired / sim.n
            err = (sim.target_p - frac) / max(sim.target_p, 1e-9)
            sim.gain = float(np.clip(sim.gain * np.exp(p.gain_adapt * np.clip(err, -1, 1)), *p.gain_bounds))
            sim.activity.push(spikes_np)
            results.append(spikes_np)

        return results


# --- ModernGL Compute Shader Backend (Vendor-Neutral GPU) --------------------
_moderngl_available = False
try:
    import moderngl
    _moderngl_available = True
except ImportError:
    moderngl = None


class _GLContextBinder:
    """Helper to bind ModernGL standalone contexts across worker threads (EGL, GLX, WGL)."""

    def __init__(self) -> None:
        self._libegl = None
        self._libgl = None
        self._libwgl = None
        import ctypes
        import sys

        try:
            libegl = ctypes.CDLL("libEGL.so.1")
            libegl.eglGetCurrentContext.restype = ctypes.c_void_p
            libegl.eglGetCurrentDisplay.restype = ctypes.c_void_p
            libegl.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            libegl.eglMakeCurrent.restype = ctypes.c_bool
            self._libegl = libegl
        except Exception:
            pass

        try:
            libgl = ctypes.CDLL("libGL.so.1")
            libgl.glXGetCurrentContext.restype = ctypes.c_void_p
            libgl.glXGetCurrentDisplay.restype = ctypes.c_void_p
            libgl.glXMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            libgl.glXMakeCurrent.restype = ctypes.c_bool
            self._libgl = libgl
        except Exception:
            pass

        if sys.platform == "win32":
            try:
                opengl32 = ctypes.windll.opengl32
                opengl32.wglGetCurrentContext.restype = ctypes.c_void_p
                opengl32.wglGetCurrentDC.restype = ctypes.c_void_p
                opengl32.wglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                opengl32.wglMakeCurrent.restype = ctypes.c_bool
                self._libwgl = opengl32
            except Exception:
                pass

    def capture_current(self):
        if self._libegl:
            ctx = self._libegl.eglGetCurrentContext()
            if ctx:
                return ("egl", self._libegl.eglGetCurrentDisplay(), ctx)
        if self._libgl:
            ctx = self._libgl.glXGetCurrentContext()
            if ctx:
                return ("glx", self._libgl.glXGetCurrentDisplay(), ctx)
        if self._libwgl:
            ctx = self._libwgl.wglGetCurrentContext()
            if ctx:
                return ("wgl", self._libwgl.wglGetCurrentDC(), ctx)
        return None

    def make_current(self, handle) -> None:
        if not handle:
            return
        kind = handle[0]
        if kind == "egl":
            self._libegl.eglMakeCurrent(handle[1], None, None, handle[2])
        elif kind == "glx":
            self._libgl.glXMakeCurrent(handle[1], 0, handle[2])
        elif kind == "wgl":
            self._libwgl.wglMakeCurrent(handle[1], handle[2])

    def release_current(self, handle) -> None:
        if not handle:
            return
        kind = handle[0]
        if kind == "egl":
            self._libegl.eglMakeCurrent(handle[1], None, None, None)
        elif kind == "glx":
            self._libgl.glXMakeCurrent(handle[1], 0, None)
        elif kind == "wgl":
            self._libwgl.wglMakeCurrent(handle[1], None)


_context_binder = _GLContextBinder()


def _gl_compute_available() -> tuple[bool, str]:
    """Check if OpenGL 4.3+ compute shaders are supported on this system."""
    if not _moderngl_available:
        return False, "ModernGL is not installed"
    try:
        ctx = moderngl.create_context(standalone=True)
        ver = ctx.version_code / 100.0
        if ctx.version_code < 430:
            ctx.release()
            return False, f"OpenGL {ver:.1f} < 4.3 (Compute shaders not supported)"
        renderer = ctx.info.get("GL_RENDERER", "OpenGL GPU")
        ctx.release()
        return True, f"OpenGL {ver:.1f}: {renderer}"
    except Exception as e:
        return False, f"OpenGL context creation failed: {e}"


_SPMV_COMPUTE_SHADER = """
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer BRowPtr { uint row_ptr[]; };
layout(std430, binding = 1) readonly buffer BColInd { uint col_ind[]; };
layout(std430, binding = 2) readonly buffer BValues { float values[]; };
layout(std430, binding = 5) readonly buffer BSpikes { uint spikes[]; };
layout(std430, binding = 8) writeonly buffer BISyn   { float i_syn[]; };
uniform uint num_neurons;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= num_neurons) return;
    uint start = row_ptr[i];
    uint end = row_ptr[i + 1];
    float sum = 0.0;
    for (uint idx = start; idx < end; ++idx) {
        if (spikes[col_ind[idx]] != 0u) {
            sum += values[idx];
        }
    }
    i_syn[i] = sum;
}
"""

_LIF_COMPUTE_SHADER = """
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 3) buffer BV       { float v[]; };
layout(std430, binding = 4) buffer BRefr    { int refr[]; };
layout(std430, binding = 5) buffer BSpikes  { uint spikes[]; };
layout(std430, binding = 6) readonly buffer BNoise   { float noise[]; };
layout(std430, binding = 7) readonly buffer BSens    { float sensory[]; };
layout(std430, binding = 8) readonly buffer BISyn    { float i_syn[]; };

uniform uint num_neurons;
uniform uint noise_offset;
uniform float gain;
uniform float bias;
uniform float ext_gain;
uniform float leak_decay;
uniform float leak_reset;
uniform float v_reset;
uniform float v_thresh;
uniform int refr_steps;
uniform int has_sensory;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= num_neurons) return;

    float drive = i_syn[i] * gain + bias + noise[noise_offset + i];
    if (has_sensory != 0) {
        drive += sensory[i] * ext_gain;
    }

    float v_val = v[i] * leak_decay + leak_reset + drive;
    int r = refr[i];
    if (r > 0) {
        v_val = v_reset;
        r -= 1;
    }

    if (v_val >= v_thresh) {
        spikes[i] = 1u;
        v[i] = v_reset;
        refr[i] = refr_steps;
    } else {
        spikes[i] = 0u;
        v[i] = v_val;
        refr[i] = r;
    }
}
"""


class GLBackend(SimBackend):
    """ModernGL compute shader backend: vendor-neutral GPU acceleration using OpenGL 4.3+ compute shaders and SSBOs.
    Runs on AMD, NVIDIA, and Intel GPUs across Linux and Windows without requiring PyTorch."""

    name = "gl"

    def __init__(self, sim: Any) -> None:
        super().__init__(sim)
        self.ctx = None
        self.cs_spmv = None
        self.cs_lif = None
        self.buf_rowptr = None
        self.buf_colind = None
        self.buf_values = None
        self.buf_v = None
        self.buf_refr = None
        self.buf_spikes = None
        self.buf_noise = None
        self.buf_sens = None
        self.buf_isyn = None
        self.num_groups = (self.n + 63) // 64
        self._noise_id = None
        self._thread_id = None

    def setup(self) -> None:
        if not _moderngl_available:
            raise RuntimeError("ModernGL is not installed")
        ok, dev = _gl_compute_available()
        if not ok:
            raise RuntimeError(dev)
        self.device_name = dev
        # Initialize resources on the calling thread
        self._init_device()

    def _init_device(self) -> None:
        import threading
        self._thread_id = threading.get_ident()
        if self.ctx is not None:
            try:
                self.ctx.release()
            except Exception:
                pass
            self.ctx = None

        try:
            self.ctx = moderngl.create_context(standalone=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create ModernGL context: {e}")

        if self.ctx.version_code < 430:
            ver = self.ctx.version_code / 100.0
            self.ctx.release()
            self.ctx = None
            raise RuntimeError(f"OpenGL {ver:.1f} does not support compute shaders (OpenGL 4.3+ required)")

        csr = self.sim.W_csr
        self.buf_rowptr = self.ctx.buffer(csr.indptr.astype(np.uint32).tobytes())
        self.buf_colind = self.ctx.buffer(csr.indices.astype(np.uint32).tobytes())
        self.buf_values = self.ctx.buffer(csr.data.astype(np.float32).tobytes())

        self.buf_v = self.ctx.buffer(reserve=self.n * 4)
        self.buf_refr = self.ctx.buffer(reserve=self.n * 4)
        self.buf_spikes = self.ctx.buffer(reserve=self.n * 4)
        self.buf_noise = self.ctx.buffer(self.sim._noise.astype(np.float32).tobytes())
        self._noise_id = id(self.sim._noise)
        self.buf_sens = self.ctx.buffer(reserve=self.n * 4)
        self.buf_isyn = self.ctx.buffer(reserve=self.n * 4)

        self.cs_spmv = self.ctx.compute_shader(_SPMV_COMPUTE_SHADER)
        self.cs_lif = self.ctx.compute_shader(_LIF_COMPUTE_SHADER)

        self.buf_rowptr.bind_to_storage_buffer(0)
        self.buf_colind.bind_to_storage_buffer(1)
        self.buf_values.bind_to_storage_buffer(2)
        self.buf_v.bind_to_storage_buffer(3)
        self.buf_refr.bind_to_storage_buffer(4)
        self.buf_spikes.bind_to_storage_buffer(5)
        self.buf_noise.bind_to_storage_buffer(6)
        self.buf_sens.bind_to_storage_buffer(7)
        self.buf_isyn.bind_to_storage_buffer(8)

        self.cs_spmv["num_neurons"] = self.n
        self.cs_lif["num_neurons"] = self.n
        self._init_uniforms()
        self.sync_from_host()

    def _ensure_thread(self) -> None:
        import threading
        if self.ctx is None or self._thread_id != threading.get_ident():
            self._init_device()

    def _init_uniforms(self) -> None:
        p = self.sim.p
        sim = self.sim
        self.cs_lif["bias"] = float(p.bias)
        self.cs_lif["ext_gain"] = float(p.ext_gain)
        self.cs_lif["leak_decay"] = float(1.0 - sim.leak)
        self.cs_lif["leak_reset"] = float(sim.leak * p.v_reset) if p.v_reset else 0.0
        self.cs_lif["v_reset"] = float(p.v_reset)
        self.cs_lif["v_thresh"] = float(p.v_thresh)
        self.cs_lif["refr_steps"] = int(p.refractory_steps)

    def on_weights_changed(self) -> None:
        self._ensure_thread()
        if self.buf_values is not None:
            self.buf_values.write(self.sim.W_csr.data.astype(np.float32).tobytes())

    def sync_to_host(self) -> None:
        self._ensure_thread()
        if self.buf_v is not None:
            self.sim.v[:] = np.frombuffer(self.buf_v.read(), dtype=np.float32)
            self.sim.refr[:] = np.frombuffer(self.buf_refr.read(), dtype=np.int32).astype(np.int16)
            self.sim.spikes[:] = np.frombuffer(self.buf_spikes.read(), dtype=np.uint32).astype(bool)

    def sync_from_host(self) -> None:
        self._ensure_thread()
        if self.buf_v is not None:
            self.buf_v.write(self.sim.v.astype(np.float32).tobytes())
            self.buf_refr.write(self.sim.refr.astype(np.int32).tobytes())
            self.buf_spikes.write(self.sim.spikes.astype(np.uint32).tobytes())
            if self._noise_id != id(self.sim._noise):
                self.buf_noise.write(self.sim._noise.astype(np.float32).tobytes())
                self._noise_id = id(self.sim._noise)

    def step(self, sensory_input: np.ndarray | None = None) -> np.ndarray:
        self._ensure_thread()
        sim = self.sim
        if self._noise_id != id(sim._noise):
            self.buf_noise.write(sim._noise.astype(np.float32).tobytes())
            self._noise_id = id(sim._noise)

        # 1. Sparse SpMV propagation
        self.cs_spmv.run(self.num_groups)
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

        # 2. Sensory input upload if active
        if sensory_input is not None:
            self.buf_sens.write(np.ascontiguousarray(sensory_input, dtype=np.float32).tobytes())
            self.cs_lif["has_sensory"] = 1
        else:
            self.cs_lif["has_sensory"] = 0

        # 3. LIF elementwise update
        off = int(sim.rng.integers(0, sim._noise.size - self.n))
        self.cs_lif["noise_offset"] = off
        self.cs_lif["gain"] = float(sim.gain)
        self.cs_lif.run(self.num_groups)
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)

        # 4. Read spikes
        spikes_raw = np.frombuffer(self.buf_spikes.read(), dtype=np.uint32)
        spikes = spikes_raw.astype(bool)
        sim.spikes = spikes
        return spikes


def detect_available_backends() -> dict[str, str]:
    """The backends that can run here, mapped to human-readable device names."""
    avail = {"cpu": CPUBackend.device_name}
    if _numba_available:
        avail["numba"] = NumbaBackend.device_name
    ok, gl_dev = _gl_compute_available()
    if ok:
        avail["gl"] = gl_dev
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
    build of PyTorch, 'gl' for OpenGL Compute, or 'cpu' after a fallback), and that is what gets recorded everywhere."""
    choice = (backend_choice or "auto").lower()
    b: SimBackend | None = None
    if choice == "auto":
        if _torch_gpu_kind():
            b = _try(lambda: TorchBackend(sim, "cuda:0"), "PyTorch GPU")
        if b is None and _moderngl_available:
            b = _try(lambda: GLBackend(sim), "OpenGL Compute")
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
    elif choice == "gl":
        if not _moderngl_available:
            log.warning("gl requested but ModernGL is not installed; using the CPU backend")
        else:
            b = _try(lambda: GLBackend(sim), "OpenGL Compute")
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
