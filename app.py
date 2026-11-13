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

RELU = nn.ReLU()

def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1

device = "cuda"

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

@spaces.GPU
def enhance(filepath, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0, exit_layer=8, look_ahead_frames=0):
    """Enhance one file.

    Returns ``(input_path, output_path, spectrogram_figure, detailed_log)``.
    The fourth value is the run log: pipeline stages with timings, GPU memory,
    scan throughput (iterations/s) and a per-block profile. It is also written
    to ``logs/inference/`` with a JSON sidecar. On failure the log is returned
    instead of raising, so the reason (and the OOM mitigation hints) reaches
    the UI.
    """
    # Gradio pasa la etiqueta completa del dropdown; convertirla al ID interno del registro.
    if variant_id in VARIANT_LABEL_TO_ID:
        variant_id = VARIANT_LABEL_TO_ID[variant_id]

    entry = VARIANT_REGISTRY.get(variant_id)
    is_realtime_family = bool(entry and entry.get("family") == "realtime")
    mode = "single-pass (whole file)" if entry and not entry.get("chunked", True) else "chunked overlap-add"
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
                logger.warn(
                    f"the low-pass ({lp_sr} Hz) and target ({target_sr} Hz) sliders are not "
                    f"applied by this pipeline; the model still sees {noisy_sr} Hz audio. "
                    "Resample the file before uploading to get the memory/speed benefit."
                )

            SE_model = _get_model(variant_id, logger=logger)

            profiler = ModuleProfiler(SE_model)
            profiler.start()
            model_seconds = 0.0

            # 2. Configuración de chunking: Standard usa overlap-add; RealTime pasa single-pass
            if entry.get("chunked", True):
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
                    if predicted_peak_mb > mem["free_mb"]:
                        logger.warn(
                            f"predicted peak ({predicted_peak_mb / 1024:.2f} GB) exceeds free "
                            f"memory ({mem['free_mb'] / 1024:.2f} GB). Expect an OOM on Linux or "
                            "a silent spill into shared system memory on Windows (which can be "
                            "10-20x slower). Split the file, use mono, or resample."
                        )
                        for line in OOM_HINTS:
                            logger.info("  " + line)

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

                with logger.stage(
                    "model_forward",
                    note=f"single pass over {audio_seconds:.2f} s, {geom['t_scan']} scan frames",
                ) as rec:
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
                print(f"[Timer] Inferencia single-pass completada", flush=True)
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

with gr.Blocks() as demo:
    gr.Markdown(ABOUT)
    gr.Markdown("**Note 1**: For bandwidth extension, the performance may be affected by the characteristics of the input data, particularly the cutoff pattern. A simple solution is to apply low-pass filtering beforehand.")
    gr.Markdown("**Note 2**: When processing long input audio, out-of-memory (OOM) errors may occur. The **RealTime** variants run a single forward pass over the whole file, so memory grows with duration and sample rate. Split long recordings, use mono, resample to 16-48 kHz, or pick the **Standard** variant, which processes in chunks. Every run now prints a **Detailed Processing Log** with per-stage timings, GPU memory and scan iterations/s, and saves it to `logs/inference/`.")
    gr.Markdown("**Note 3**: The low-pass and target sample-rate sliders are currently not applied by the pipeline; resample the file before uploading if you need the memory/speed benefit (the log warns you when they are set).")

    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="Input Audio (WAV, FLAC, OGG, MP3)", type="filepath")
            batch_input_files = gr.File(
                file_count="multiple",
                label="Batch Input Files (WAV, FLAC, OGG, MP3) - optional batch processing",
                interactive=True,
            )
            variant_dropdown = gr.Dropdown(
                label="Model Variant",
                choices=[
                    "Standard (30 bloques)",
                    "Laughter · preserva risa (30 bloques)",
                    "BF16 · menor tamaño (30 bloques)",
                    "RealTime · baja latencia (12 bloques)",
                    "Whisper · diálogos bajos (12 bloques)",
                ],
                value="Standard (30 bloques)",
            )
            target_sampling_rate = gr.Slider(
                minimum=0, maximum=48000, value=0, step=1000,
                label="(Optional) Target sampling rate (0 = off). Lower = faster processing but loses high-frequency detail.",
                interactive=True,
            )
            low_pass_sampling_rate = gr.Slider(
                minimum=0, maximum=48000, value=0, step=1000,
                label="(Optional) Pre low-pass filter rate (0 = off). Lower = faster but removes more high frequencies.",
                interactive=True,
            )
            chunk_size_s = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="Chunk size (seconds). Smaller = lower memory but more overhead per chunk.",
                interactive=True,
            )
            overlap_portion = gr.Slider(
                minimum=0.0, maximum=0.9, value=0.5, step=0.1,
                label="Overlap portion. Lower = faster but may cause join artifacts between chunks.",
                interactive=True,
            )
            window_type = gr.Dropdown(
                choices=["Hann", "Hamming"], value="Hann",
                label="Window type. Hann = smoother; Hamming = sharper edges.",
                interactive=True,
            )
            normalization = gr.Slider(
                minimum=0.0, maximum=1.0, value=1.0, step=0.1,
                label="Normalization. Lower = quieter output. No speed impact.",
                interactive=True,
            )
            exit_layer_slider = gr.Slider(
                minimum=3, maximum=12, value=8, step=1,
                label="Exit layer (RealTime/Whisper only). Lower = faster but worse quality.",
                interactive=False,
            )
            look_ahead_frames_slider = gr.Slider(
                minimum=0, maximum=2, value=0, step=1,
                label="Look ahead frames (RealTime/Whisper only). Lower = faster but less future context.",
                interactive=False,
            )

    # Variant-dependent enable/disable for RealTime/Whisper sliders.
    # Gradio passes the full dropdown label; resolve it to the internal variant ID
    # via VARIANT_LABEL_TO_ID and check the family flag.
    variant_dropdown.change(
        fn=lambda variant: (
            gr.update(interactive=VARIANT_LABEL_TO_ID.get(variant, "standard") in ("realtime", "whisper")),
            gr.update(interactive=VARIANT_LABEL_TO_ID.get(variant, "standard") in ("realtime", "whisper")),
        ),
        inputs=[variant_dropdown],
        outputs=[exit_layer_slider, look_ahead_frames_slider],
    )

    enhance_btn = gr.Button("Enhance")
    batch_enhance_btn = gr.Button("Batch Enhance All Files")
    open_output_folder_btn = gr.Button("Open Output Folder")
    open_logs_folder_btn = gr.Button("Open Logs Folder")
    with gr.Row():
        input_audio_player = gr.Audio(label="Original Input Audio", type="filepath")
        output_audio = gr.Audio(label="Enhanced Audio", type="filepath")
    batch_log = gr.Textbox(label="Batch Processing Log", lines=5, interactive=False)
    detail_log = gr.Textbox(
        label="Detailed Processing Log (per-stage timings, GPU memory, scan iterations/s)",
        lines=20,
        max_lines=40,
        interactive=False,
    )
    plot_output = gr.Plot(label="Spectrograms")

    def unified_enhance(audio_path, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0, exit_layer=8, look_ahead_frames=0):
        return enhance(audio_path, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer, look_ahead_frames)

    def batch_enhance_files(batch_files, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0, exit_layer=8, look_ahead_frames=0):
        if batch_files is None:
            return None, None, None, "No batch files selected.", ""
        results = []
        messages = []
        details = []
        fig = None
        for f in batch_files:
            name = os.path.basename(f.name)
            try:
                original, enhanced_path, fig, detail = enhance(f.name, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer, look_ahead_frames)
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
        fn=unified_enhance,
        inputs=[input_audio, low_pass_sampling_rate, target_sampling_rate, variant_dropdown, chunk_size_s, overlap_portion, window_type, normalization, exit_layer_slider, look_ahead_frames_slider],
        outputs=[input_audio_player, output_audio, plot_output, detail_log]
    )

    batch_enhance_btn.click(
        fn=batch_enhance_files,
        inputs=[batch_input_files, low_pass_sampling_rate, target_sampling_rate, variant_dropdown, chunk_size_s, overlap_portion, window_type, normalization, exit_layer_slider, look_ahead_frames_slider],
        outputs=[input_audio_player, output_audio, plot_output, batch_log, detail_log]
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
