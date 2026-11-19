"""LiteRT (Metal) backend for RE-USE on Apple Silicon.

Offline 16 kHz mono speech enhancement with the fixed-chunk graphs published by
litert-community. The graph consumes [1, 2, T, 321] (log1p|X|, angle X) and
returns [1, 3, T, 321] (compressed magnitude m, phase x_r, x_i); the STFT,
inverse STFT and overlap-add run on the host, following the model cards:

    https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT
    https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT

This is an offline file enhancer. It is NOT a streaming microphone filter.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

SR = 16000
N_FFT = 640
HOP = 80
BINS = N_FFT // 2 + 1
MODEL_DIR = "models_litert"

# Periodic Hann window, matching the graph's training preprocessing.
WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)

# key -> (hf repo, fastload filename, main filename, UI label)
MODEL_SPECS = {
    "standard": (
        "litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT",
        "reuse_16k_2s_fastload_fp32.tflite",
        "reuse_16k_2s_fp32.tflite",
        "Standard",
    ),
    "laughter": (
        "litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT",
        "reuse_laughter_16k_2s_fastload_fp32.tflite",
        "reuse_laughter_16k_2s_fp32.tflite",
        "Laughter (preserva risas)",
    ),
}

GRAPH_LABELS = {
    "fastload": "Fastload · arranque rápido (~10 s)",
    "main": "Main · máxima calidad (~131 s de carga)",
}


def model_path(key: str, graph: str = "fastload") -> str:
    """Return the local path of a downloaded graph, or raise FileNotFoundError."""
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key: {key}")
    if graph not in ("fastload", "main"):
        raise KeyError(f"Unknown graph: {graph}")
    filename = MODEL_SPECS[key][1 if graph == "fastload" else 2]
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(
            f"Missing model file: {path}. Re-run Install to download the LiteRT graphs."
        )
    return path


def list_available_models() -> dict:
    """Report which graph files are already downloaded."""
    return {
        key: {
            label: {
                graph: os.path.isfile(os.path.join(MODEL_DIR, MODEL_SPECS[key][idx]))
                and os.path.getsize(os.path.join(MODEL_DIR, MODEL_SPECS[key][idx])) > 0
                for idx, graph in ((1, "fastload"), (2, "main"))
            }
            for label in (spec[3],)
        }
        for key, spec in MODEL_SPECS.items()
    }


def read_audio(path: str):
    """Read any audio file as 16 kHz mono float32, resampling when needed."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] > 1:
        x = x.mean(axis=1)
    else:
        x = x[:, 0]
    if sr != SR:
        import resampy

        x = resampy.resample(x, sr, SR)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def stft(chunk: np.ndarray) -> np.ndarray:
    """[n] -> [1, 2, T, 321]: log1p|X| and angle(X), centered with reflect padding."""
    padded = np.pad(chunk, N_FFT // 2, mode="reflect")
    frames = sliding_window_view(padded, N_FFT)[::HOP] * WINDOW
    spec = np.fft.rfft(frames, axis=-1)
    return np.stack([np.log1p(np.abs(spec)), np.angle(spec)])[None].astype(np.float32)


def istft(y: np.ndarray, n: int) -> np.ndarray:
    """[1, 3, T, 321] (m, x_r, x_i) -> n samples, as in the model cards."""
    m, xr, xi = y[0]
    mag = np.expm1(np.maximum(m, 0))
    mag[(mag == 0).mean(axis=1) > 0.5] = 0
    h = np.hypot(xr, xi)
    safe = np.where(h > 0, h, 1)
    phase = np.where(h > 0, (xr + 1j * xi) / safe, 1)
    frames = np.fft.irfft(mag * phase, n=N_FFT, axis=-1) * WINDOW

    length = N_FFT + HOP * (len(frames) - 1)
    out = np.zeros(length, np.float32)
    env = np.zeros(length, np.float32)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(len(frames))[:, None]
    np.add.at(out, idx.ravel(), frames.ravel())
    env_frames = np.broadcast_to(WINDOW ** 2, frames.shape)
    np.add.at(env, idx.ravel(), env_frames.ravel())

    s = slice(N_FFT // 2, N_FFT // 2 + n)
    return (out[s] / env[s]).astype(np.float32)


def chunk_starts(n: int, chunk: int, hop: int) -> list:
    """Chunk starts as in NVIDIA's inference_chunk.py: last chunk aligned to end."""
    if n <= chunk:
        return [0]
    return list(range(0, n - chunk, hop)) + [n - chunk]


@dataclass
class EnhanceResult:
    samples: np.ndarray
    seconds: float
    chunks: int
    frames: int
    chunk_samples: int
    accel: str


class ReuseLiteRT:
    """Thin wrapper over the ai-edge-litert CompiledModel API."""

    def __init__(self, model, frames: int, accel: str):
        self.model = model
        self.frames = frames
        self.chunk = HOP * (frames - 1)
        self.hop = self.chunk * 7 // 8 if self.chunk >= 2 * SR else self.chunk // 2
        self.accel = accel
        self.inputs = model.create_input_buffers(0)
        self.outputs = model.create_output_buffers(0)
        try:
            self.out_size = model.get_output_buffer_requirements(0, 0)["buffer_size"] // 4
        except Exception:
            self.out_size = frames * BINS * 3

    @classmethod
    def load(cls, key: str, graph: str = "fastload", accel: str = "gpu") -> "ReuseLiteRT":
        from ai_edge_litert.compiled_model import (
            CompiledModel,
            CpuOptions,
            HardwareAccelerator,
            Options,
        )

        path = model_path(key, graph)
        actual = accel
        try:
            options = (
                Options(hardware_accelerators=HardwareAccelerator.GPU)
                if accel == "gpu"
                else Options(
                    hardware_accelerators=HardwareAccelerator.CPU,
                    cpu_options=CpuOptions(num_threads=4),
                )
            )
            model = CompiledModel.from_file(path, options=options)
        except Exception as exc:
            if accel != "gpu":
                raise
            # Metal delegate can be unavailable on some setups: retry on CPU.
            options = Options(
                hardware_accelerators=HardwareAccelerator.CPU,
                cpu_options=CpuOptions(num_threads=4),
            )
            model = CompiledModel.from_file(path, options=options)
            actual = "cpu"

        try:
            frames = model.get_input_tensor_details("serving_default")["args_0"]["shape"][2]
        except Exception:
            frames = 401
        return cls(model, int(frames), actual)

    def enhance(self, x: np.ndarray) -> EnhanceResult:
        started = time.time()
        n = len(x)
        padded = np.pad(x, (0, max(0, self.chunk - n)))
        win = np.maximum(
            0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.chunk) / self.chunk), 1e-7
        ).astype(np.float32)

        out = np.zeros(len(padded), np.float32)
        wsum = np.zeros(len(padded), np.float32)
        starts = chunk_starts(len(padded), self.chunk, self.hop)
        for start in starts:
            self.inputs[0].write(stft(padded[start : start + self.chunk]))
            self.model.run_by_index(0, self.inputs, self.outputs)
            y = self.outputs[0].read(self.out_size, np.float32).reshape(1, 3, self.frames, -1)
            out[start : start + self.chunk] += istft(y, self.chunk) * win
            wsum[start : start + self.chunk] += win

        return EnhanceResult(
            samples=(out / wsum)[:n],
            seconds=time.time() - started,
            chunks=len(starts),
            frames=self.frames,
            chunk_samples=self.chunk,
            accel=self.accel,
        )

    def close(self) -> None:
        for buffer in self.inputs + self.outputs:
            try:
                buffer.close()
            except Exception:
                pass
        try:
            self.model.close()
        except Exception:
            pass


def write_wav(path: str, samples: np.ndarray) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, samples, SR, subtype="FLOAT")
    return path
