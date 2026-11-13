# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import subprocess
import tempfile
import importlib
import spaces
import gradio as gr
import soundfile as sf
import time
import math
import contextlib

CUDA_HOME = "/cuda-image/usr/local/cuda-13.0"

from reuse_runtime import (
    bind_selective_scan_reference,
    install_fast_selective_scan,
    patch_mamba_selective_scan,
)
from reuse_profiler import (
    MEMORY_OVERHEAD_FACTOR,
    InferenceLogger,
    ModuleProfiler,
    estimate_scan_workspace_mb,
    gpu_memory,
    model_scan_shape,
    predict_peak_mb,
    scan_time_frames,
    stft_geometry,
)

# The pinned mamba tag hard-imports the compiled selective_scan_cuda extension.
# Make that import optional, then route mamba_simple to the reference kernel.
patch_mamba_selective_scan()
bind_selective_scan_reference()

ABOUT = """\
# RE-USE: A universal speech enhancement model for diverse degradations, sampling rates, and languages.
Upload or record a noisy clip, then click **Enhance** to listen to the result and view its spectrogram.
(ref: https://huggingface.co/spaces/rc19477/Speech_Enhancement_Mamba)
"""

import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "RE-USE requires an NVIDIA GPU with a working CUDA 12.8 driver. "
        "torch.cuda.is_available() is False. Re-run Install or check your NVIDIA driver."
    )

import torchaudio
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
import numpy as np
from models.stfts import mag_phase_stft, mag_phase_istft
from utils.util import load_config, pad_or_trim_to_match
from huggingface_hub import hf_hub_download
from reuse_streaming import build_streaming_model, run_streaming_frames

RELU = nn.ReLU()

def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1

device = "cuda"

# Streaming runs the selective scan with a batch of one frame, where the
# reference kernel's Python loop dominates. Swap in the chunked kernel there;
# it self-tests against the reference and leaves it bound on any mismatch.
install_fast_selective_scan()

# ---------------------------------------------------------------------------
# Model variant registry (Phase 2 tracer slice)
# Maps variant id -> recipe YAML, model class/module, weight source, etc.
# ---------------------------------------------------------------------------
VARIANT_REGISTRY = {
    # --- Gama Standard (30 bloques Mamba, inferencia chunked) ---
    "standard": {
        "label": "Standard (30 bloques)",
        "recipe": "recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml",
        "model_class": "SEMamba",
        "model_module": "models.generator_SEMamba_time_d4",
        "weight_source": "hf_hub",
        "weight_repo": "nvidia/NVIDIA-NemotronLabs-RE-USE",
        "chunked": True,
        "family": "standard",
        # The generator pads +2 time frames and DenseEncoder uses a (1, 3)
        # conv with stride (4, 2): the Mamba stack sees T / 4 frames.
        "encoder_time_stride": 4,
        "encoder_time_kernel": 1,
        "encoder_time_pad": 2,
    },
    "laughter": {
        "label": "Laughter · preserva risa (30 bloques)",
        "recipe": "recipes/USEMamba_30x1_laughter_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml",
        "model_class": "SEMamba",
        "model_module": "models.generator_SEMamba_time_d4",
        "weight_source": "hf_hub",
        "weight_repo": "nvidia/NVIDIA-NemotronLabs-RE-USE-laughter",
        "chunked": True,
        "family": "standard",
        "encoder_time_stride": 4,
        "encoder_time_kernel": 1,
        "encoder_time_pad": 2,
    },
    "bf16": {
        "label": "BF16 · menor tamaño (30 bloques)",
        "recipe": "recipes/USEMamba_30x1_bf16_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml",
        "model_class": "SEMamba",
        "model_module": "models.generator_SEMamba_time_d4",
        "weight_source": "hf_hub",
        "weight_repo": "giangndm/NVIDIA-NemotronLabs-RE-USE-bf16",
        "chunked": True,
        "family": "standard",
        "encoder_time_stride": 4,
        "encoder_time_kernel": 1,
        "encoder_time_pad": 2,
    },
    # --- Gama RealTime (12 bloques Mamba, inferencia single-pass) ---
    "realtime": {
        "label": "RealTime · baja latencia (12 bloques)",
        "recipe": "recipes/USEMamba_12x1_lr_00002_norm_05_vq_067_nfft_320_hop_160_NRIR_012_pha_0005_com_04_early_005_release_random_layer_GAN_longer_1k.yaml",
        "model_class": "SEMamba_decoder_list",
        "model_module": "models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv",
        "weight_source": "hf_hub",
        "weight_repo": "nvidia/Real-time_RE-USE",
        "chunked": False,
        "extra_ui_params": ["exit_layer", "look_ahead_frames"],
        "exit_layer_default": 8,
        "look_ahead_frames_default": 0,
        "family": "realtime",
        # The encoder pads +2 time frames, then a (3, 3) conv with stride
        # (1, 2) consumes that padding: the Mamba stack sees T frames.
        "encoder_time_stride": 1,
        "encoder_time_kernel": 3,
        "encoder_time_pad": 2,
    },
    "whisper": {
        "label": "Whisper · diálogos bajos (12 bloques)",
        "recipe": "recipes/USEMamba_12x1_whisper_lr_00002_norm_05_vq_067_nfft_320_hop_160_NRIR_012_pha_0005_com_04_early_005_release_random_layer_GAN_longer_1k.yaml",
        "model_class": "SEMamba_decoder_list",
        "model_module": "models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv",
        "weight_source": "hf_hub",
        "weight_repo": "Malfaro43/Realtime-REUSE-whisper-finetune",
        "chunked": False,
        "extra_ui_params": ["exit_layer", "look_ahead_frames"],
        "exit_layer_default": 8,
        "look_ahead_frames_default": 0,
        "family": "realtime",
        "encoder_time_stride": 1,
        "encoder_time_kernel": 3,
        "encoder_time_pad": 2,
    },
}

# Mapa de etiquetas del dropdown de Gradio -> IDs de VARIANT_REGISTRY.
# Gradio pasa la etiqueta completa del elemento seleccionado, no la clave interna.
VARIANT_LABEL_TO_ID = {entry["label"]: vid for vid, entry in VARIANT_REGISTRY.items()}

# Load full recipe configs per variant (used for model construction + audio params).
# The Standard variant keeps the original `cfg1` name/shape for full backward
# compatibility with the Phase 1 `from_pretrained(cfg=cfg1)` call path.
_VARIANT_FULL_CFG = {}
for _vid, _entry in VARIANT_REGISTRY.items():
    recipe_path = _entry["recipe"]
    if os.path.exists(recipe_path):
        _VARIANT_FULL_CFG[_vid] = load_config(recipe_path)
    else:
        print(f"Warning: recipe config '{recipe_path}' for variant '{_vid}' not found on disk; variant '{_vid}' will be unavailable until the recipe file is added.", flush=True)

if "standard" not in _VARIANT_FULL_CFG:
    raise SystemExit("Standard variant recipe config is missing; cannot start RE-USE.")
cfg1 = _VARIANT_FULL_CFG["standard"]
n_fft, hop_size, win_size = cfg1['stft_cfg']['n_fft'], cfg1['stft_cfg']['hop_size'], cfg1['stft_cfg']['win_size']
compress_factor = cfg1['model_cfg']['compress_factor']
sampling_rate = cfg1['stft_cfg']['sampling_rate']

_USE_MODELS = {}

OOM_HINTS = [
    "The RealTime variants run a single forward pass over the whole file, so the",
    "selective-scan workspace grows linearly with duration and frequency width:",
    "  workspace ~= 2 tensors * 4 B * t_scan * f_enc * d_inner * d_state",
    "  (t_scan is the coded time grid the Mamba stack sees: T for RealTime, T/4",
    "   for Standard, so both run at 50 frames per second of audio.)",
    "It is not a model-size problem (13.5M params); it is the pure-Python scan",
    "kernel materialising deltaA / deltaB_u for the full utterance.",
    "Mitigations:",
    "  1. Split the input into clips of a few seconds, or use the Standard",
    "     variant, which already runs chunk-wise with overlap-add.",
    "  2. Convert to mono: stereo is currently fed as a batch of 2, doubling",
    "     every activation.",
    "  3. Resample to 16 kHz or 8 kHz before uploading. f_enc scales with the",
    "     sample rate (48 kHz -> 481 bins, 16 kHz -> 161, 8 kHz -> 81), so the",
    "     scan workspace drops ~3x at 16 kHz and ~6x at 8 kHz.",
    "  4. Lower 'Exit layer' to 3-4: fewer Mamba blocks, same peak per block but",
    "     less fragmentation and a shorter run.",
]


def _get_model(variant_id, logger=None):
    """Load and cache a model per variant id.

    Looks up VARIANT_REGISTRY, loads the recipe YAML, dynamically imports the
    model class module, instantiates the class, loads weights per source
    (from_pretrained for standard, torch.load for realtime), calls .to(device)
    and .eval(), and caches per variant id.
    """
    if variant_id not in _USE_MODELS:
        entry = VARIANT_REGISTRY[variant_id]
        cfg = _VARIANT_FULL_CFG[variant_id]
        source = entry.get("weight_repo") or entry.get("weight_path", "local weights")
        if logger:
            logger.info(f"loading model '{variant_id}' from {source} (first use, downloads if missing)")
        print(f"Loading {variant_id} model (first time, may take a while)...", flush=True)
        with (logger.stage("load_model", note=variant_id) if logger else contextlib.nullcontext()):
            mod = importlib.import_module(entry["model_module"])
            model_class = getattr(mod, entry["model_class"])
            if entry["weight_source"] == "hf_hub":
                model = model_class.from_pretrained(entry.get("weight_repo", "nvidia/NVIDIA-NemotronLabs-RE-USE"), cfg=cfg).to(device)
            elif entry["weight_source"] == "local_pth":
                model = model_class(cfg).to(device)
                state = torch.load(entry["weight_path"], map_location=device)
                model.load_state_dict(state)
            else:
                raise ValueError(f"Unknown weight_source: {entry['weight_source']}")
            model.eval()
        _USE_MODELS[variant_id] = model
        n_params = sum(p.numel() for p in model.parameters())
        if logger:
            logger.info(
                f"model ready: {n_params / 1e6:.2f}M parameters, "
                f"{n_params * 4 / (1024 * 1024):.0f} MB weights (fp32), "
                f"{len(model.TSMamba)} Mamba blocks"
            )
        print(f"{variant_id} model loaded.", flush=True)
    elif logger:
        logger.info(f"reusing cached model '{variant_id}'")
    return _USE_MODELS[variant_id]

@contextlib.contextmanager
def block_progress(model, total_blocks, label="model"):
    """Print per-Mamba-block progress to the terminal during a forward pass.

    Hooks ``model.TSMamba[:total_blocks]`` so each completed block reports its
    index, cumulative time and an ETA. A ``torch.cuda.synchronize()`` is issued
    per block to keep the timing honest, which is cheap relative to a block.
    """
    blocks = list(getattr(model, "TSMamba", []))[: max(int(total_blocks or 0), 0)]
    if not blocks:
        yield
        return

    total = len(blocks)
    state = {"done": 0}
    start = time.perf_counter()
    handles = []

    def make_hook():
        def hook(_module, _inp, _out):
            state["done"] += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            done = state["done"]
            avg = elapsed / done
            remaining = max(total - done, 0) * avg
            pct = done / total * 100.0
            print(
                f"[reuse][{label}] block {done}/{total} "
                f"({pct:5.1f}%)  {elapsed:7.2f}s elapsed  "
                f"~{remaining:7.2f}s remaining  {avg * 1000:7.1f} ms/block",
                flush=True,
            )
        return hook

    try:
        for block in blocks:
            handles.append(block.register_forward_hook(make_hook()))
        yield
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass


@spaces.GPU
def enhance(filepath, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0, exit_layer=8, look_ahead_frames=0, streaming=False):
    """Enhance one file.

    Returns ``(input_path, output_path, spectrogram_figure, detailed_log)``.
    The fourth value is the run log: pipeline stages with timings, GPU memory,
    scan throughput (iterations/s) and a per-block profile. It is also written
    to ``logs/inference/`` with a JSON sidecar. On failure the log is returned
    instead of raising, so the reason (and the OOM mitigation hints) reaches
    the UI.

    ``lp_sr`` / ``target_sr`` reproduce the original NVIDIA app's bandwidth
    extension: when a target rate is set the file is resampled to ``lp_sr``
    first (the low-pass step) and then to ``target_sr``, which is the rate the
    model runs at and the output is written at. ``streaming`` selects the
    frame-by-frame RealTime path, which keeps memory flat over the file length.
    """
    # Gradio pasa la etiqueta completa del dropdown; convertirla al ID interno del registro.
    if variant_id in VARIANT_LABEL_TO_ID:
        variant_id = VARIANT_LABEL_TO_ID[variant_id]

    entry = VARIANT_REGISTRY.get(variant_id)
    is_realtime_family = bool(entry and entry.get("family") == "realtime")
    if entry and not entry.get("chunked", True):
        mode = "streaming (frame by frame)" if streaming else "single-pass (whole file)"
    else:
        mode = "chunked overlap-add"
    logger = InferenceLogger(run_name=f"{variant_id}_enhance")
    profiler = None

    meta = {
        "input": filepath,
        "variant": variant_id,
        "mode": mode,
    }
    try:
        header = sf.info(filepath)
        meta["sample_rate"] = f"{header.samplerate} Hz"
        meta["channels"] = header.channels
        meta["duration"] = f"{header.frames / header.samplerate:.2f} s"
    except Exception as exc:
        meta["input"] = f"{filepath} (header unreadable: {exc})"
    if is_realtime_family:
        meta["exit_layer"] = exit_layer
        meta["look_ahead"] = look_ahead_frames
    logger.begin(**meta)

    try:
        with torch.no_grad():
            # 1. Lectura robusta con soundfile (evita totalmente el error de torchcodec.dll)
            with logger.stage("read_audio"):
                _data, noisy_sr = sf.read(filepath, dtype='float32', always_2d=True)
                Noisy_wav = torch.from_numpy(_data.T.copy())
                original_noisy_wav = Noisy_wav.clone()
                original_sr = noisy_sr

            # 1b. Optional bandwidth extension, with the same semantics as the
            # original NVIDIA app: a target rate enables the sliders, the
            # low-pass rate is applied first, then the target rate, and the
            # model runs (and the output is written) at the target rate.
            lp_hz = int(lp_sr or 0)
            target_hz = int(target_sr or 0)
            if target_hz:
                with logger.stage("resample", note=f"{noisy_sr} Hz -> {target_hz} Hz") as rec:
                    if lp_hz:
                        Noisy_wav = torch.from_numpy(
                            librosa.resample(
                                Noisy_wav.cpu().numpy(),
                                orig_sr=noisy_sr,
                                target_sr=lp_hz,
                                res_type="kaiser_best",
                            )
                        )
                        noisy_sr = lp_hz
                        logger.info(
                            f"low-pass: resampled to {lp_hz} Hz before the model, which removes "
                            "content above that rate instead of faking it"
                        )
                    Noisy_wav = torch.from_numpy(
                        librosa.resample(
                            Noisy_wav.cpu().numpy(),
                            orig_sr=noisy_sr,
                            target_sr=target_hz,
                            res_type="kaiser_best",
                        )
                    )
                    noisy_sr = target_hz
                    logger.info(
                        f"target: processing and writing at {target_hz} Hz "
                        f"({Noisy_wav.shape[1] / noisy_sr:.2f} s)"
                    )
                meta["resampled_to"] = f"{target_hz} Hz"
            elif lp_hz:
                logger.warn(
                    f"low-pass rate ({lp_hz} Hz) ignored without a target rate: the original app "
                    "only applies it when a target rate is set. Set both, or neither."
                )

            if entry is None:
                raise ValueError(f"Unknown variant id '{variant_id}'.")
            if variant_id not in _VARIANT_FULL_CFG:
                raise ValueError(
                    f"Variant '{variant_id}' is not available: recipe config '{entry['recipe']}' "
                    "is missing on disk. Add the recipe YAML or select the Standard variant."
                )
            cfg = _VARIANT_FULL_CFG[variant_id]
            v_n_fft = cfg['stft_cfg']['n_fft']
            v_hop_size = cfg['stft_cfg']['hop_size']
            v_win_size = cfg['stft_cfg']['win_size']
            v_compress_factor = cfg['model_cfg']['compress_factor']
            v_sampling_rate = cfg['stft_cfg']['sampling_rate']

            n_fft_scaled = make_even(v_n_fft * noisy_sr // v_sampling_rate)
            hop_size_scaled = make_even(v_hop_size * noisy_sr // v_sampling_rate)
            win_size_scaled = make_even(v_win_size * noisy_sr // v_sampling_rate)

            n_channels, n_samples = Noisy_wav.shape
            audio_seconds = n_samples / noisy_sr
            geom = stft_geometry(
                cfg,
                noisy_sr,
                n_samples,
                time_stride=entry.get("encoder_time_stride", 1),
                time_kernel=entry.get("encoder_time_kernel", 1),
                time_pad=entry.get("encoder_time_pad", 0),
            )
            d_inner, d_state = model_scan_shape(cfg)

            logger.info(
                f"audio: {n_samples} samples @ {noisy_sr} Hz, {n_channels} channel(s), "
                f"{audio_seconds:.2f} s"
            )
            logger.info(
                f"stft (scaled to input rate): n_fft={n_fft_scaled} hop={hop_size_scaled} "
                f"win={win_size_scaled} -> {geom['freq_bins']} bins, {geom['fps']:.1f} frames/s"
            )
            logger.info(
                f"encoder grid: STFT T={geom['t_frames']} -> scan T={geom['t_scan']} x F={geom['f_enc']} "
                f"(time stride {geom['time_stride']}; d_inner={d_inner}, d_state={d_state})"
            )

            # -- preflight warnings ------------------------------------------------
            if n_channels > 1:
                if entry.get("chunked", True):
                    detail = (
                        f"they are processed one channel at a time, so the runtime is "
                        f"multiplied by {n_channels}"
                    )
                else:
                    detail = (
                        f"they are processed as a batch of {n_channels} inside a single "
                        f"forward pass, so every activation is multiplied by {n_channels}"
                    )
                logger.warn(
                    f"input has {n_channels} channels; {detail}. The model expects mono audio."
                )
            if noisy_sr > 48000:
                logger.warn(
                    f"input is {noisy_sr} Hz (> 48 kHz): the frequency axis grows with the "
                    "sample rate, so memory grows too. Resample to 16-48 kHz."
                )
            if lp_sr or target_sr:
                logger.info(
                    f"resampling request: low-pass {lp_hz or 'off'} Hz -> target "
                    f"{target_hz or 'off'} Hz; the frequency axis (and therefore memory) "
                    f"scales with the rate the model runs at ({noisy_sr} Hz)"
                )

            SE_model = _get_model(variant_id, logger=logger)

            profiler = ModuleProfiler(SE_model)
            profiler.start()
            model_seconds = 0.0
            streaming_used = False

            # 2. Configuración de chunking: Standard usa overlap-add; RealTime pasa single-pass
            if entry.get("chunked", True):
                if streaming:
                    logger.warn(
                        "the Streaming box only applies to the RealTime/Whisper variants; this "
                        "variant already processes in chunks, so the checkbox is ignored"
                    )
                chunk_size = int(chunk_size_s * noisy_sr)  # slider-driven chunk size
                hop_length = int(overlap_portion * chunk_size)  # slider-driven overlap
                if window_type == "Hamming":
                    window = torch.clamp(torch.hamming_window(chunk_size).to(device), min=1e-7)
                else:
                    window = torch.clamp(torch.hann_window(chunk_size).to(device), min=1e-7)

                num_chunks = max(1, math.ceil((Noisy_wav.shape[1] - chunk_size) / hop_length) + 1)
                logger.info(
                    f"chunking: {num_chunks} chunk(s) of {chunk_size} samples "
                    f"({chunk_size / noisy_sr:.2f} s), hop {hop_length} samples "
                    f"({overlap_portion:.0%} overlap), {window_type} window"
                )
                # The chunk size, not the file length, sets the memory peak here.
                chunk_t_frames = chunk_size // hop_size_scaled + 1
                chunk_t_scan = scan_time_frames(
                    chunk_t_frames,
                    geom["time_stride"],
                    entry.get("encoder_time_kernel", 1),
                    entry.get("encoder_time_pad", 0),
                )
                chunk_workspace_mb = estimate_scan_workspace_mb(
                    chunk_t_scan, geom["f_enc"], d_inner, d_state
                )
                chunk_peak_mb = predict_peak_mb(
                    chunk_t_scan, geom["f_enc"], d_inner, d_state
                )
                logger.info(
                    f"per-chunk scan workspace: {chunk_workspace_mb / 1024:.2f} GB "
                    f"(estimated peak {chunk_peak_mb / 1024:.2f} GB) for scan "
                    f"T={chunk_t_scan} x F={geom['f_enc']} (STFT T={chunk_t_frames}); "
                    "chunk size is the memory knob in this mode"
                )
                mem = gpu_memory()
                if mem and chunk_peak_mb > mem["free_mb"]:
                    logger.warn(
                        f"predicted per-chunk peak ({chunk_peak_mb / 1024:.2f} GB) exceeds "
                        f"free memory ({mem['free_mb'] / 1024:.2f} GB): lower 'Chunk size', "
                        "use mono, or resample before uploading."
                    )

                with logger.stage("chunk_setup"):
                    Noisy_wav = Noisy_wav.to(device)
                    audio_enhanced = torch.zeros_like(Noisy_wav).to(device)
                    window_sum = torch.zeros_like(Noisy_wav).to(device)

                t_inf = time.time()

                # 3. Bucle por canales y fragmentos
                for c in range(Noisy_wav.shape[0]):
                    noisy_wav = Noisy_wav[c:c+1, :]
                    for i in range(num_chunks):
                        label = (
                            f"ch{c + 1} chunk {i + 1}/{num_chunks}"
                            if n_channels > 1
                            else f"chunk {i + 1}/{num_chunks}"
                        )
                        with logger.stage(label) as rec:
                            if i < num_chunks - 1:
                                noisy_wav_chunk = noisy_wav[:, i * hop_length : i * hop_length + chunk_size]
                            else:
                                noisy_wav_chunk = noisy_wav[:, -chunk_size:]

                            t_stft = time.perf_counter()
                            noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
                                noisy_wav_chunk,
                                n_fft=n_fft_scaled,
                                hop_size=hop_size_scaled,
                                win_size=win_size_scaled,
                                compress_factor=v_compress_factor,
                                center=True,
                                addeps=False
                            )
                            torch.cuda.synchronize()
                            t_model = time.perf_counter()

                            total_blocks = exit_layer if is_realtime_family else len(SE_model.TSMamba)
                            with block_progress(SE_model, total_blocks, label=f"{label}"):
                                if is_realtime_family:
                                    amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha, layer_use=exit_layer, number_ahead=look_ahead_frames)
                                else:
                                    amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha)
                            torch.cuda.synchronize()
                            t_istft = time.perf_counter()
                            model_seconds += t_istft - t_model

                            # Limpieza de artefactos
                            mag = torch.expm1(RELU(amp_g))
                            zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
                            amp_g[:, :, (zero_portion > 0.5)[0]] = 0

                            audio_g = mag_phase_istft(amp_g, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, v_compress_factor)
                            audio_g = pad_or_trim_to_match(noisy_wav_chunk.detach(), audio_g, pad_value=1e-8)

                            # Acumulación Overlap-Add con ventana de Hann
                            if i < num_chunks - 1:
                                audio_enhanced[c:c+1, i * hop_length : i * hop_length + chunk_size] += audio_g * window[0:audio_g.shape[1]]
                                window_sum[c:c+1, i * hop_length : i * hop_length + chunk_size] += window[0:audio_g.shape[1]]
                            else:
                                audio_enhanced[c:c+1, -chunk_size:] += audio_g * window[0:audio_g.shape[1]]
                                window_sum[c:c+1, -chunk_size:] += window[0:audio_g.shape[1]]

                            torch.cuda.empty_cache()
                            t_done = time.perf_counter()
                            if rec is not None:
                                rec["note"] = (
                                    f"ch={c} at {(i * hop_length) / noisy_sr:.2f}s | "
                                    f"stft {(t_model - t_stft) * 1000:.0f}ms "
                                    f"model {(t_istft - t_model) * 1000:.0f}ms "
                                    f"istft+ola {(t_done - t_istft) * 1000:.0f}ms"
                                )

                # Normalización final dividiendo por la suma de ventanas
                with logger.stage("overlap_add_normalize"):
                    audio_enhanced = audio_enhanced / torch.clamp(window_sum * normalization, min=1e-7)

                torch.cuda.synchronize()
                print(f"[Timer] Inferencia por chunks completada: {time.time() - t_inf:.4f}s", flush=True)
                assert audio_enhanced.shape == Noisy_wav.shape, audio_enhanced.shape
            else:
                # Single-pass RealTime inference (no chunking)
                predicted_mb = estimate_scan_workspace_mb(
                    geom["t_scan"], geom["f_enc"], d_inner, d_state, batch=n_channels
                )
                predicted_peak_mb = predict_peak_mb(
                    geom["t_scan"], geom["f_enc"], d_inner, d_state, batch=n_channels
                )
                mem = gpu_memory()
                logger.info(
                    f"predicted selective-scan workspace: {predicted_mb / 1024:.2f} GB "
                    f"(peak of a single Mamba block pass, {d_inner} x {d_state} state)"
                )
                logger.info(
                    f"predicted total peak: {predicted_peak_mb / 1024:.2f} GB "
                    f"(scan workspace x {MEMORY_OVERHEAD_FACTOR} measured activation overhead)"
                )
                if mem:
                    logger.info(
                        f"gpu memory before inference: {mem['free_mb'] / 1024:.2f} GB free "
                        f"of {mem['total_mb'] / 1024:.2f} GB"
                    )

                # Streaming keeps only one frame of context in cached causal-conv
                # and Mamba states, so memory no longer grows with duration. It is
                # also slower per second of audio, so it only takes over by itself
                # when the single pass would not fit.
                use_streaming = bool(streaming)
                if use_streaming:
                    logger.info(
                        "mode: streaming requested; memory stays flat over the file length "
                        "at the cost of a slower model pass"
                    )
                elif mem and predicted_peak_mb > mem["free_mb"]:
                    use_streaming = True
                    logger.warn(
                        f"predicted single-pass peak ({predicted_peak_mb / 1024:.2f} GB) exceeds "
                        f"free memory ({mem['free_mb'] / 1024:.2f} GB): switching to frame-by-frame "
                        "streaming, which keeps memory flat. Tick the Streaming box to force it "
                        "even when the single pass fits."
                    )
                streaming_used = use_streaming

                with logger.stage("to_device"):
                    Noisy_wav = Noisy_wav.to(device)
                with logger.stage(
                    "stft",
                    note=f"{geom['freq_bins']} bins x {geom['t_frames']} frames",
                ):
                    noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
                        Noisy_wav,
                        n_fft=n_fft_scaled,
                        hop_size=hop_size_scaled,
                        win_size=win_size_scaled,
                        compress_factor=v_compress_factor,
                        center=True,
                        addeps=False
                    )
                # `com` is never consumed by the model; drop it before the forward
                # pass, which is where the memory pressure is.
                del noisy_com
                torch.cuda.empty_cache()

                if use_streaming:
                    with logger.stage(
                        "streaming_setup",
                        note=f"{geom['f_enc']} frequency bins of state, "
                             f"look-ahead {look_ahead_frames} frame(s)",
                    ):
                        stream_model = build_streaming_model(SE_model)
                        stream_model.reset(geom["f_enc"])
                        logger.info(
                            f"streaming: {geom['t_frames']} frames of "
                            f"{1000.0 / geom['fps']:.0f} ms, exit layer {exit_layer}, "
                            f"{geom['f_enc']} cached frequency states per block; "
                            "each frame reuses the weights of the loaded model"
                        )

                    with logger.stage(
                        "streaming_frames",
                        note=f"{geom['t_frames']} frames one by one",
                    ) as rec:
                        torch.cuda.reset_peak_memory_stats()
                        amp_g, pha_g, frame_seconds = run_streaming_frames(
                            stream_model,
                            noisy_mag,
                            noisy_pha,
                            exit_layer,
                            look_ahead_frames,
                            progress=True,
                            progress_every=max(1, geom["t_frames"] // 20),
                            progress_label="streaming",
                        )
                        torch.cuda.synchronize()
                    model_seconds += rec["seconds"] if rec else 0.0
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    del stream_model

                    ordered = sorted(frame_seconds)
                    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
                    logger.info(
                        f"streaming latency per frame: first {frame_seconds[0] * 1000:.1f} ms "
                        f"(includes warm-up), mean {sum(frame_seconds) / len(frame_seconds) * 1000:.1f} ms, "
                        f"median {ordered[len(ordered) // 2] * 1000:.1f} ms, p95 {p95 * 1000:.1f} ms"
                    )
                    logger.info(
                        f"streaming peak GPU memory: {peak_mb:.1f} MB "
                        f"(single pass would need {predicted_peak_mb / 1024:.2f} GB); "
                        "memory is independent of the file length"
                    )
                    logger.set("streaming", True)
                    logger.set("streaming_peak_mb", round(peak_mb, 1))
                else:
                    with logger.stage(
                        "model_forward",
                        note=f"single pass over {audio_seconds:.2f} s, {geom['t_scan']} scan frames",
                    ) as rec:
                        with block_progress(SE_model, exit_layer, label="model_forward"):
                            amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha, layer_use=exit_layer, number_ahead=look_ahead_frames)
                        torch.cuda.synchronize()
                    model_seconds += rec["seconds"] if rec else 0.0

                with logger.stage("mask_postprocess"):
                    mag = torch.expm1(RELU(amp_g))
                    zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
                    amp_g[:, :, (zero_portion > 0.5)[0]] = 0
                with logger.stage("istft"):
                    audio_enhanced = mag_phase_istft(amp_g, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, v_compress_factor)
                with logger.stage("pad_trim"):
                    audio_enhanced = pad_or_trim_to_match(Noisy_wav.detach(), audio_enhanced, pad_value=1e-8)
                torch.cuda.synchronize()
                print("[Timer] Inferencia RealTime completada", flush=True)
                assert audio_enhanced.shape == Noisy_wav.shape, audio_enhanced.shape

            profiler.stop()

            # 4. Estadísticas del modelo (iteraciones del scan y velocidad)
            stats = profiler.stats()
            if model_seconds > 0:
                iter_per_s = stats["scan_iterations"] / model_seconds
                state_per_s = stats["state_updates"] / model_seconds
                logger.info(
                    f"scan throughput: {stats['scan_iterations']:,} scan iterations "
                    f"({stats['state_updates']:,} state updates) in {stats['scan_calls']} "
                    f"kernel calls over {model_seconds:.2f} s of model time "
                    f"-> {iter_per_s:,.0f} scan_iter/s, {state_per_s:,.0f} state_updates/s"
                )
                if streaming_used:
                    logger.info(
                        "note: in streaming the temporal Mamba runs through mamba.step(), which "
                        "bypasses the forward hooks, so these scan numbers only cover the "
                        "frequency path"
                    )
                logger.info(
                    f"speed: {model_seconds:.2f} s of model time for {audio_seconds:.2f} s of audio "
                    f"-> RTF {model_seconds / audio_seconds:.3f} "
                    f"({audio_seconds / model_seconds:.2f}x realtime)"
                )
                logger.set("scan_iterations", stats["scan_iterations"])
                logger.set("scan_iter_per_s", iter_per_s)
                logger.set("rtf", model_seconds / audio_seconds if audio_seconds else None)
            profiler.log_to(logger, forward_seconds=model_seconds)

            # 5. Guardar únicamente el archivo mejorado en la carpeta output con sufijo único
            with logger.stage("save_wav"):
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)

                # Extraemos el nombre del archivo original sin la extensión
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                enh_filename = f"{base_name}_enhanced.wav"
                enh_path = os.path.join(output_dir, enh_filename)

                # Guardamos solo el resultado procesado
                sf.write(enh_path, audio_enhanced.cpu().numpy().T, noisy_sr)

            # Generación de espectrogramas
            with logger.stage("spectrogram"):
                fig, axs = plt.subplots(1, 2, figsize=(16, 4))
                D_noisy = librosa.stft(original_noisy_wav[0].cpu().numpy(), n_fft=512, hop_length=256)
                S_noisy = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)
                librosa.display.specshow(S_noisy, sr=original_sr, hop_length=256, x_axis="time", y_axis="hz", ax=axs[0], vmax=0)
                axs[0].set_title("Noisy Spectrogram")

                D_clean = librosa.stft(audio_enhanced[0].cpu().numpy(), n_fft=512, hop_length=256)
                S_clean = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
                librosa.display.specshow(S_clean, sr=noisy_sr, hop_length=256, x_axis="time", y_axis="hz", ax=axs[1], vmax=0)
                axs[1].set_title("Enhanced Spectrogram")
                plt.tight_layout()

            logger.set("status", "ok")
            log_text = logger.finish(summary=f"OK -> {enh_path}")
            # Devolvemos la ruta original para el reproductor, la nueva ruta limpia y el gráfico
            return filepath, enh_path, fig, log_text

    except Exception as exc:
        message = str(exc)
        logger.set("status", "failed")
        logger.set("error", message)
        logger.warn(f"{type(exc).__name__}: {message}")
        oom = "out of memory" in message.lower() or "OutOfMemory" in type(exc).__name__
        if oom:
            logger.info("CUDA OOM while running this file. Diagnosis and mitigations:")
            for line in OOM_HINTS:
                logger.info("  " + line)
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        log_text = logger.finish(summary=f"FAILED: {type(exc).__name__}: {message}")
        print(f"Error crítico en inferencia: {message}", flush=True)
        return filepath, None, None, log_text
    finally:
        if profiler is not None:
            profiler.stop()

with gr.Blocks(title="RE-USE Speech Enhancement") as demo:
    gr.Markdown(ABOUT)

    with gr.Accordion("Notas y consejos", open=False):
        gr.Markdown(
            "- **Bandwidth extension:** el resultado puede depender del patrón de corte del audio; "
            "un pre-filtrado paso-bajo puede ayudar.\n"
            "- **Memoria (RealTime/Whisper):** el modo single-pass crece con duración y tasa de muestreo. "
            "Activa **Streaming** para memoria plana (más lento) o se activará solo si el pico previsto "
            "no cabe en VRAM. **Standard** ya procesa en chunks (el tamaño de chunk es la palanca de memoria).\n"
            "- **Muestreo:** *Target sampling rate* hace que el modelo procese y escriba a esa tasa; "
            "*Pre low-pass* se aplica antes y solo si hay target (0 = desactivado). Bajar la tasa reduce el eje "
            "de frecuencia y la memoria.\n"
            "- Cada ejecución escribe un **log detallado** (tiempos por etapa, memoria GPU, scan/RTF, perfil por bloque) "
            "en `logs/inference/`."
        )

    active_tab = gr.State("single")

    with gr.Row():
        variant_dropdown = gr.Dropdown(
            label="Model variant",
            choices=[
                "Standard (30 bloques)",
                "Laughter · preserva risa (30 bloques)",
                "BF16 · menor tamaño (30 bloques)",
                "RealTime · baja latencia (12 bloques)",
                "Whisper · diálogos bajos (12 bloques)",
            ],
            value="Standard (30 bloques)",
            scale=3,
        )
        enhance_btn = gr.Button("Enhance", variant="primary", scale=1)

    with gr.Tabs() as input_tabs:
        with gr.Tab("Audio", id="single") as tab_single:
            input_audio = gr.Audio(label="Input audio (WAV, FLAC, OGG, MP3)", type="filepath")
        with gr.Tab("Lote (batch)", id="batch") as tab_batch:
            batch_input_files = gr.File(
                file_count="multiple",
                label="Archivos (WAV, FLAC, OGG, MP3)",
                interactive=True,
            )
            batch_log = gr.Textbox(label="Batch log", lines=4, max_lines=12, interactive=False)

    tab_single.select(lambda: "single", None, active_tab)
    tab_batch.select(lambda: "batch", None, active_tab)

    with gr.Accordion("Ajustes avanzados", open=False):
        with gr.Row():
            target_sampling_rate = gr.Slider(
                minimum=0, maximum=48000, value=0, step=1000,
                label="Target sampling rate (Hz, 0 = off)",
                info="Procesa y guarda a esta tasa (menos memoria si es menor). Activa el pre low-pass.",
                interactive=True,
            )
            low_pass_sampling_rate = gr.Slider(
                minimum=0, maximum=48000, value=0, step=1000,
                label="Pre low-pass (Hz, 0 = off)",
                info="Se aplica antes del target; solo tiene efecto si target > 0 (BWE).",
                interactive=True,
            )
        with gr.Row():
            chunk_size_s = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="Chunk size (s, Standard)",
                info="Menor = menos memoria, más solapamiento por chunk.",
                interactive=True,
            )
            overlap_portion = gr.Slider(
                minimum=0.0, maximum=0.9, value=0.5, step=0.1,
                label="Overlap (Standard)",
                info="Menor = más rápido, posibles artefactos en uniones.",
                interactive=True,
            )
        with gr.Row():
            window_type = gr.Dropdown(
                choices=["Hann", "Hamming"], value="Hann",
                label="Window (Standard)",
                info="Hann: más suave; Hamming: bordes más definidos.",
                interactive=True,
            )
            normalization = gr.Slider(
                minimum=0.1, maximum=1.0, value=1.0, step=0.1,
                label="Normalization",
                info="Menor = salida más alta (más volumen).",
                interactive=True,
            )
        with gr.Row():
            exit_layer_slider = gr.Slider(
                minimum=3, maximum=12, value=8, step=1,
                label="Exit layer (RealTime/Whisper)",
                info="Menor = más rápido, menor calidad.",
                interactive=False,
            )
            look_ahead_frames_slider = gr.Slider(
                minimum=0, maximum=2, value=0, step=1,
                label="Look-ahead frames (RealTime/Whisper)",
                info="Menor = menos latencia, menos contexto futuro.",
                interactive=False,
            )
        streaming_checkbox = gr.Checkbox(
            value=False,
            label="Streaming (RealTime/Whisper)",
            info="Frame a frame con estados cacheados: memoria plana, pasada más lenta. Se activa solo si el single-pass no cabría en VRAM.",
            interactive=False,
        )

    # Variant-dependent enable/disable for RealTime/Whisper sliders.
    # Gradio passes the full dropdown label; resolve it to the internal variant ID
    # via VARIANT_LABEL_TO_ID and check the family flag.
    variant_dropdown.change(
        fn=lambda variant: (
            gr.update(interactive=VARIANT_LABEL_TO_ID.get(variant, "standard") in ("realtime", "whisper")),
            gr.update(interactive=VARIANT_LABEL_TO_ID.get(variant, "standard") in ("realtime", "whisper")),
            gr.update(interactive=VARIANT_LABEL_TO_ID.get(variant, "standard") in ("realtime", "whisper")),
        ),
        inputs=[variant_dropdown],
        outputs=[exit_layer_slider, look_ahead_frames_slider, streaming_checkbox],
    )

    with gr.Row():
        input_audio_player = gr.Audio(label="Original", type="filepath")
        output_audio = gr.Audio(label="Enhanced", type="filepath")

    with gr.Tabs():
        with gr.Tab("Log detallado"):
            detail_log = gr.Textbox(
                label="Per-stage timings, GPU memory, scan/RTF",
                lines=8,
                max_lines=30,
                interactive=False,
            )
        with gr.Tab("Espectrogramas"):
            plot_output = gr.Plot(label="Spectrograms")

    with gr.Row():
        with gr.Column():
            open_output_folder_btn = gr.Button("Open Output Folder")
        with gr.Column():
            open_logs_folder_btn = gr.Button("Open Logs Folder")

    def unified_enhance(audio_path, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0, exit_layer=8, look_ahead_frames=0, streaming=False):
        return enhance(audio_path, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer, look_ahead_frames, streaming)

    def batch_enhance_files(batch_files, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0, exit_layer=8, look_ahead_frames=0, streaming=False):
        if batch_files is None:
            return None, None, None, "No batch files selected.", ""
        results = []
        messages = []
        details = []
        fig = None
        for f in batch_files:
            name = os.path.basename(f.name)
            try:
                original, enhanced_path, fig, detail = enhance(f.name, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer, look_ahead_frames, streaming)
                if enhanced_path:
                    results.append(enhanced_path)
                    messages.append(f"OK   {name} -> {os.path.basename(enhanced_path)}")
                else:
                    messages.append(f"FAIL {name} -> see the detailed log below")
                details.append(f"===== {name} =====\n{detail}")
            except Exception as e:
                messages.append(f"FAIL {name}: {str(e)}")
        log_text = "\n".join(messages)
        detail_text = "\n\n".join(details)
        return results[0] if results else None, results[-1] if results else None, fig if results else None, log_text, detail_text

    def process_active(active, audio_path, batch_files, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0, exit_layer=8, look_ahead_frames=0, streaming=False):
        if active == "batch":
            return batch_enhance_files(batch_files, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer, look_ahead_frames, streaming)
        if not audio_path:
            return None, None, None, "", "No input audio selected."
        original, enhanced_path, fig, detail = unified_enhance(audio_path, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer, look_ahead_frames, streaming)
        return original, enhanced_path, fig, "", detail

    def open_output_folder():
        if sys.platform == "win32":
            os.startfile("output")
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", "output"])
        else:  # macOS
            subprocess.Popen(["open", "output"])
        return "Output folder opened."

    def open_logs_folder():
        log_dir = os.path.abspath("logs/inference")
        os.makedirs(log_dir, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(log_dir)
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", log_dir])
        else:  # macOS
            subprocess.Popen(["open", log_dir])
        return f"Logs folder opened: {log_dir}"

    enhance_btn.click(
        fn=process_active,
        inputs=[
            active_tab, input_audio, batch_input_files,
            low_pass_sampling_rate, target_sampling_rate, variant_dropdown,
            chunk_size_s, overlap_portion, window_type, normalization,
            exit_layer_slider, look_ahead_frames_slider, streaming_checkbox,
        ],
        outputs=[input_audio_player, output_audio, plot_output, batch_log, detail_log],
    )

    open_output_folder_btn.click(
        fn=open_output_folder,
        inputs=None,
        outputs=None,
    )

    open_logs_folder_btn.click(
        fn=open_logs_folder,
        inputs=None,
        outputs=None,
    )

if __name__ == "__main__":
    demo.queue().launch(share=False)
