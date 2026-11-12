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

CUDA_HOME = "/cuda-image/usr/local/cuda-13.0"

from reuse_runtime import (
    bind_selective_scan_reference,
    patch_mamba_selective_scan,
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
    "standard": {
        "label": "Standard",
        "recipe": "recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml",
        "model_class": "SEMamba",
        "model_module": "models.generator_SEMamba_time_d4",
        "weight_source": "hf_hub",
        "weight_repo": "nvidia/NVIDIA-NemotronLabs-RE-USE",
        "chunked": True,
    },
    "realtime": {
        "label": "RealTime",
        "recipe": "recipes/USEMamba_12x1_lr_00002_norm_05_vq_067_nfft_320_hop_160_NRIR_012_pha_0005_com_04_early_005_release_random_layer_GAN_longer_1k.yaml",
        "model_class": "SEMamba_decoder_list",
        "model_module": "models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv",
        "weight_source": "local_pth",
        "weight_path": "exp/12x1_lr_00002_norm_05_vq_065_nfft_320_hop_160_NRIR_012_pha_0005_com_04_peak_release_random_layer_ahead_sep_conv2/g_01010000.pth",
        "chunked": False,
        "extra_ui_params": ["exit_layer", "look_ahead_frames"],
        "exit_layer_default": 8,
        "look_ahead_frames_default": 0,
    },
}

# Load full recipe configs per variant (used for model construction + audio params).
# The Standard variant keeps the original `cfg1` name/shape for full backward
# compatibility with the Phase 1 `from_pretrained(cfg=cfg1)` call path.
_VARIANT_FULL_CFG = {}
for _vid, _entry in VARIANT_REGISTRY.items():
    _VARIANT_FULL_CFG[_vid] = load_config(_entry["recipe"])

cfg1 = _VARIANT_FULL_CFG["standard"]
n_fft, hop_size, win_size = cfg1['stft_cfg']['n_fft'], cfg1['stft_cfg']['hop_size'], cfg1['stft_cfg']['win_size']
compress_factor = cfg1['model_cfg']['compress_factor']
sampling_rate = cfg1['stft_cfg']['sampling_rate']

_USE_MODELS = {}

def _get_model(variant_id):
    """Load and cache a model per variant id.

    Looks up VARIANT_REGISTRY, loads the recipe YAML, dynamically imports the
    model class module, instantiates the class, loads weights per source
    (from_pretrained for standard, torch.load for realtime), calls .to(device)
    and .eval(), and caches per variant id.
    """
    if variant_id not in _USE_MODELS:
        entry = VARIANT_REGISTRY[variant_id]
        cfg = _VARIANT_FULL_CFG[variant_id]
        print(f"Loading {variant_id} model (first time, may take a while)...", flush=True)
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
        print(f"{variant_id} model loaded.", flush=True)
    return _USE_MODELS[variant_id]

@spaces.GPU
def enhance(filepath, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0):
    try:
        with torch.no_grad():
            # 1. Lectura robusta con soundfile (evita totalmente el error de torchcodec.dll)
            _data, noisy_sr = sf.read(filepath, dtype='float32', always_2d=True)
            Noisy_wav = torch.from_numpy(_data.T.copy())
            original_noisy_wav = Noisy_wav.clone()
            original_sr = noisy_sr

            entry = VARIANT_REGISTRY[variant_id]
            cfg = _VARIANT_FULL_CFG[variant_id]
            v_n_fft = cfg['stft_cfg']['n_fft']
            v_hop_size = cfg['stft_cfg']['hop_size']
            v_win_size = cfg['stft_cfg']['win_size']
            v_compress_factor = cfg['model_cfg']['compress_factor']
            v_sampling_rate = cfg['stft_cfg']['sampling_rate']

            n_fft_scaled = make_even(v_n_fft * noisy_sr // v_sampling_rate)
            hop_size_scaled = make_even(v_hop_size * noisy_sr // v_sampling_rate)
            win_size_scaled = make_even(v_win_size * noisy_sr // v_sampling_rate)

            SE_model = _get_model(variant_id)

            exit_layer = entry.get("exit_layer_default", 8)
            look_ahead_frames = entry.get("look_ahead_frames_default", 0)

            # 2. Configuración de chunking: Standard usa overlap-add; RealTime pasa single-pass
            if entry.get("chunked", True):
                chunk_size = int(chunk_size_s * noisy_sr)  # slider-driven chunk size
                hop_length = int(overlap_portion * chunk_size)  # slider-driven overlap
                if window_type == "Hamming":
                    window = torch.clamp(torch.hamming_window(chunk_size).to(device), min=1e-7)
                else:
                    window = torch.clamp(torch.hann_window(chunk_size).to(device), min=1e-7)

                Noisy_wav = Noisy_wav.to(device)
                audio_enhanced = torch.zeros_like(Noisy_wav).to(device)
                window_sum = torch.zeros_like(Noisy_wav).to(device)

                num_chunks = max(1, math.ceil((Noisy_wav.shape[1] - chunk_size) / hop_length) + 1)

                t_inf = time.time()

                # 3. Bucle por canales y fragmentos
                for c in range(Noisy_wav.shape[0]):
                    noisy_wav = Noisy_wav[c:c+1, :]
                    for i in range(num_chunks):
                        if i < num_chunks - 1:
                            noisy_wav_chunk = noisy_wav[:, i * hop_length : i * hop_length + chunk_size]
                        else:
                            noisy_wav_chunk = noisy_wav[:, -chunk_size:]

                        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
                            noisy_wav_chunk,
                            n_fft=n_fft_scaled,
                            hop_size=hop_size_scaled,
                            win_size=win_size_scaled,
                            compress_factor=v_compress_factor,
                            center=True,
                            addeps=False
                        )

                        if variant_id == "realtime":
                            amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha, layer_use=exit_layer, number_ahead=look_ahead_frames)
                        else:
                            amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha)

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

                # Normalización final dividiendo por la suma de ventanas
                audio_enhanced = audio_enhanced / torch.clamp(window_sum * normalization, min=1e-7)

                torch.cuda.synchronize()
                print(f"[Timer] Inferencia por chunks completada: {time.time() - t_inf:.4f}s", flush=True)
                assert audio_enhanced.shape == Noisy_wav.shape, audio_enhanced.shape
            else:
                # Single-pass RealTime inference (no chunking)
                Noisy_wav = Noisy_wav.to(device)
                noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
                    Noisy_wav,
                    n_fft=n_fft_scaled,
                    hop_size=hop_size_scaled,
                    win_size=win_size_scaled,
                    compress_factor=v_compress_factor,
                    center=True,
                    addeps=False
                )
                amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha, layer_use=exit_layer, number_ahead=look_ahead_frames)
                mag = torch.expm1(RELU(amp_g))
                zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
                amp_g[:, :, (zero_portion > 0.5)[0]] = 0
                audio_enhanced = mag_phase_istft(amp_g, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, v_compress_factor)
                audio_enhanced = pad_or_trim_to_match(Noisy_wav.detach(), audio_enhanced, pad_value=1e-8)
                torch.cuda.synchronize()
                print(f"[Timer] Inferencia single-pass completada", flush=True)
                assert audio_enhanced.shape == Noisy_wav.shape, audio_enhanced.shape

            # 4. Guardar únicamente el archivo mejorado en la carpeta output con sufijo único
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            # Extraemos el nombre del archivo original sin la extensión
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            enh_filename = f"{base_name}_enhanced.wav"
            enh_path = os.path.join(output_dir, enh_filename)

            # Guardamos solo el resultado procesado
            sf.write(enh_path, audio_enhanced.cpu().numpy().T, noisy_sr)

            # Generación de espectrogramas
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

            # Devolvemos la ruta original para el reproductor, la nueva ruta limpia y el gráfico
            return filepath, enh_path, fig

    except Exception as e:
        print(f"Error crítico en inferencia: {str(e)}", flush=True)
        raise e

with gr.Blocks() as demo:
    gr.Markdown(ABOUT)
    gr.Markdown("**Note 1**: For bandwidth extension, the performance may be affected by the characteristics of the input data, particularly the cutoff pattern. A simple solution is to apply low-pass filtering beforehand.")
    gr.Markdown("**Note 2**: When processing long input audio, out-of-memory (OOM) errors may occur. To address this, use the chunk-wise inference implementation provided on the Hugging Face.")

    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="Input Audio (WAV, FLAC, OGG, MP3)", type="filepath")
            variant_dropdown = gr.Dropdown(label="Model Variant", choices=["Standard", "RealTime"], value="Standard")
            target_sampling_rate = gr.Slider(minimum=8000, maximum=48000, value=0, step=1000, label="(Optional) Enter target sampling rate for bandwidth extension:", interactive=True)
            low_pass_sampling_rate = gr.Slider(minimum=8000, maximum=48000, value=0, step=1000, label="(Optional) Enter target sampling rate for pre-low-pass filtering before bandwidth extension:", interactive=True)
            chunk_size_s = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Chunk size (seconds)", interactive=True)
            overlap_portion = gr.Slider(minimum=0.0, maximum=0.9, value=0.5, step=0.1, label="Overlap portion", interactive=True)
            window_type = gr.Dropdown(choices=["Hann", "Hamming"], value="Hann", label="Window type", interactive=True)
            normalization = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.1, label="Normalization", interactive=True)

    enhance_btn = gr.Button("Enhance")
    with gr.Row():
        input_audio_player = gr.Audio(label="Original Input Audio", type="filepath")
        output_audio = gr.Audio(label="Enhanced Audio", type="filepath")
    plot_output = gr.Plot(label="Spectrograms")

    def unified_enhance(audio_path, lp_sr, target_sr, variant_id="standard", chunk_size_s=3, overlap_portion=0.5, window_type="Hann", normalization=1.0):
        return enhance(audio_path, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization)

    enhance_btn.click(
        fn=unified_enhance,
        inputs=[input_audio, low_pass_sampling_rate, target_sampling_rate, variant_dropdown, chunk_size_s, overlap_portion, window_type, normalization],
        outputs=[input_audio_player, output_audio, plot_output]
    )

if __name__ == "__main__":
    demo.queue().launch(share=False)
