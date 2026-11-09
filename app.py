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

ABOUT = """
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
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config, pad_or_trim_to_match
from huggingface_hub import hf_hub_download

RELU = nn.ReLU()

def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1

device = "cuda"
cfg1 = load_config('recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml')
n_fft, hop_size, win_size = cfg1['stft_cfg']['n_fft'], cfg1['stft_cfg']['hop_size'], cfg1['stft_cfg']['win_size']
compress_factor = cfg1['model_cfg']['compress_factor']
sampling_rate = cfg1['stft_cfg']['sampling_rate']

_USE_MODEL = None

def _get_model():
    global _USE_MODEL
    if _USE_MODEL is None:
        print("Loading model (first time, may take a while)...", flush=True)
        _USE_MODEL = SEMamba.from_pretrained("nvidia/NVIDIA-NemotronLabs-RE-USE", cfg=cfg1).to(device)
        _USE_MODEL.eval()
        print("Model loaded.", flush=True)
    return _USE_MODEL

@spaces.GPU
def enhance(filepath, lp_sr, target_sr):
    try:
        with torch.no_grad():
            # 1. Lectura robusta con soundfile (evita totalmente el error de torchcodec.dll)
            _data, noisy_sr = sf.read(filepath, dtype='float32', always_2d=True)
            Noisy_wav = torch.from_numpy(_data.T.copy())
            original_noisy_wav = Noisy_wav.clone()
            original_sr = noisy_sr

            n_fft_scaled = make_even(n_fft * noisy_sr // sampling_rate)
            hop_size_scaled = make_even(hop_size * noisy_sr // sampling_rate)
            win_size_scaled = make_even(win_size * noisy_sr // sampling_rate)

            SE_model = _get_model()

            # 2. Configuración oficial de chunking con solape (Overlap-Add)[cite: 3]
            chunk_size = int(3.0 * noisy_sr)  # Bloques de 3 segundos
            hop_length = int(0.5 * chunk_size)  # 50% de solape
            window = torch.clamp(torch.hann_window(chunk_size).to(device), min=1e-7)

            Noisy_wav = Noisy_wav.to(device)
            audio_enhanced = torch.zeros_like(Noisy_wav).to(device)
            window_sum = torch.zeros_like(Noisy_wav).to(device)
            
            num_chunks = max(1, math.ceil((Noisy_wav.shape[1] - chunk_size) / hop_length) + 1)

            t_inf = time.time()
            
            # 3. Bucle oficial por canales y fragmentos[cite: 3]
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
                        compress_factor=compress_factor,
                        center=True,
                        addeps=False
                    )
                    
                    amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha)
                    
                    # Limpieza de artefactos oficial[cite: 3]
                    mag = torch.expm1(RELU(amp_g))
                    zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
                    amp_g[:, :, (zero_portion > 0.5)[0]] = 0

                    audio_g = mag_phase_istft(amp_g, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, compress_factor)
                    audio_g = pad_or_trim_to_match(noisy_wav_chunk.detach(), audio_g, pad_value=1e-8)

                    # Acumulación Overlap-Add con ventana de Hann[cite: 3]
                    if i < num_chunks - 1:
                        audio_enhanced[c:c+1, i * hop_length : i * hop_length + chunk_size] += audio_g * window[0:audio_g.shape[1]]
                        window_sum[c:c+1, i * hop_length : i * hop_length + chunk_size] += window[0:audio_g.shape[1]]
                    else:
                        audio_enhanced[c:c+1, -chunk_size:] += audio_g * window[0:audio_g.shape[1]]
                        window_sum[c:c+1, -chunk_size:] += window[0:audio_g.shape[1]]
                    
                    torch.cuda.empty_cache()

            # Normalización final dividiendo por la suma de ventanas[cite: 3]
            audio_enhanced = audio_enhanced / torch.clamp(window_sum, min=1e-7)
            
            torch.cuda.synchronize()
            print(f"[Timer] Inferencia oficial por chunks completada: {time.time() - t_inf:.4f}s", flush=True)
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
            target_sampling_rate = gr.Textbox(label="(Optional) Enter target sampling rate for bandwidth extension:")
            low_pass_sampling_rate = gr.Textbox(label="(Optional) Enter target sampling rate for pre-low-pass filtering before bandwidth extension:")
    
    enhance_btn = gr.Button("Enhance")
    with gr.Row():
        input_audio_player = gr.Audio(label="Original Input Audio", type="filepath")
        output_audio = gr.Audio(label="Enhanced Audio", type="filepath")
    plot_output = gr.Plot(label="Spectrograms")

    def unified_enhance(audio_path, lp_sr, target_sr):
        return enhance(audio_path, lp_sr, target_sr)

    enhance_btn.click(
        fn=unified_enhance,
        inputs=[input_audio, low_pass_sampling_rate, target_sampling_rate],
        outputs=[input_audio_player, output_audio, plot_output]
    )

if __name__ == "__main__":
    demo.queue().launch(share=False)