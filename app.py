# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import shlex
import subprocess
import spaces
import gradio as gr

def install_mamba():
    subprocess.run(shlex.split("pip install https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"))
install_mamba()

ABOUT = """
# RE-USE: A universal speech enhancement model for diverse degradations, sampling rates, and languages.
Upload or record a noisy clip, then click **Enhance** to listen to the result and view its spectrogram.
(ref: https://huggingface.co/spaces/rc19477/Speech_Enhancement_Mamba)
"""

import torch
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

@spaces.GPU
def enhance(filepath, low_pass_sampling_rate, target_sampling_rate): 
    
    USE_model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=cfg1).to(device)
    USE_model.eval()

    with torch.no_grad():
        noisy_wav, noisy_sr = torchaudio.load(filepath)
        torchaudio.save("original.wav", noisy_wav.cpu(), noisy_sr)
        original_noisy_wav = noisy_wav
        original_sr = noisy_sr
        
        if target_sampling_rate != '':
            if low_pass_sampling_rate != '':
                opts = {"res_type": "kaiser_best"}
                noisy_wav = torch.tensor(librosa.resample(noisy_wav.cpu().numpy(), orig_sr=noisy_sr, target_sr=int(low_pass_sampling_rate), **opts))
                noisy_sr = int(low_pass_sampling_rate)
            opts = {"res_type": "kaiser_best"}
            noisy_wav = librosa.resample(noisy_wav.cpu().numpy(), orig_sr=noisy_sr, target_sr=int(target_sampling_rate), **opts)
            noisy_sr = int(target_sampling_rate)
            
        noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        n_fft_scaled = make_even(n_fft * noisy_sr // sampling_rate)
        hop_size_scaled = make_even(hop_size * noisy_sr // sampling_rate)
        win_size_scaled = make_even(win_size * noisy_sr // sampling_rate)

        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
            noisy_wav,
            n_fft=n_fft_scaled,
            hop_size=hop_size_scaled,
            win_size=win_size_scaled,
            compress_factor=compress_factor,
            center=True,
            addeps=False
        )
        amp_g, pha_g, _ = USE_model(noisy_mag, noisy_pha)
        # To remove "strange sweep artifact"
        mag = torch.expm1(RELU(amp_g)) # [1, F, T]
        zero_portion = torch.sum(mag==0, 1)/mag.shape[1]
        amp_g[:,:,(zero_portion>0.5)[0]] = 0
        
        audio_g = mag_phase_istft(amp_g, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, compress_factor)
        audio_g = pad_or_trim_to_match(noisy_wav.detach(), audio_g, pad_value=1e-8)  # Align lengths using epsilon padding
        assert audio_g.shape == noisy_wav.shape, audio_g.shape
        
        # write file
        torchaudio.save("enhanced.wav", audio_g.cpu(), noisy_sr)
        
        # spectrograms
        fig, axs = plt.subplots(1, 2, figsize=(16, 4))

        # noisy
        D_noisy = librosa.stft(original_noisy_wav[0].cpu().numpy(), n_fft=512, hop_length=256)
        S_noisy = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)
        librosa.display.specshow(S_noisy, sr=original_sr, hop_length=256, x_axis="time", y_axis="hz", ax=axs[0], vmax=0)
        axs[0].set_title("Noisy Spectrogram")

        # enhanced
        D_clean = librosa.stft(audio_g.cpu().numpy(), n_fft=512, hop_length=256)
        S_clean = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
        librosa.display.specshow(S_clean[0], sr=noisy_sr, hop_length=256, x_axis="time", y_axis="hz", ax=axs[1], vmax=0)
        axs[1].set_title("Enhanced Spectrogram")

        plt.tight_layout()
    return "original.wav", "enhanced.wav", fig

with gr.Blocks() as demo:
    gr.Markdown(ABOUT)
    gr.Markdown("**Note 1**: For bandwidth extension, the performance may be affected by the characteristics of the input data, particularly the cutoff pattern. A simple solution is to apply low-pass filtering beforehand.")
    gr.Markdown("**Note 2**: When processing long input audio, out-of-memory (OOM) errors may occur. To address this, use the chunk-wise inference implementation provided on the Hugging Face.")

    with gr.Row():
        with gr.Column():
            # Create Tabs to separate Audio and Video sessions
            with gr.Tabs():
                with gr.TabItem("Audio Upload"):
                    # gr.Audio works great for standard audio files
                    input_audio = gr.Audio(label="Input Audio", type="filepath")
                
                with gr.TabItem("Video Upload (.mp4, .mov)"):
                    # gr.File handles .mp4 and .mov without errors
                    input_video = gr.File(label="Input Video", file_types=[".mp4", ".mov"])

            target_sampling_rate = gr.Textbox(label="(Optional) Enter target sampling rate for bandwidth extension:")
            low_pass_sampling_rate = gr.Textbox(label="(Optional) Enter target sampling rate for pre-low-pass filtering before bandwidth extension:")
    
            # Helper to unify the input: we use a hidden state to store which one was used
            active_input = gr.State()
    enhance_btn = gr.Button("Enhance")
    with gr.Row():
        input_audio_player = gr.Audio(label="Original Input Audio", type="filepath")
        output_audio = gr.Audio(label="Enhanced Audio", type="filepath")
    plot_output = gr.Plot(label="Spectrograms")

    # This function determines which input (audio tab or video tab) to send to your model
    def unified_enhance(audio_path, video_path, lp_sr, target_sr):
        # Determine which path is valid (the one from the active tab)
        # Note: input_video returns a file object, so we get its .name
        final_path = audio_path if audio_path else video_path
        return enhance(final_path, lp_sr, target_sr)

    enhance_btn.click(
        fn=unified_enhance,
        inputs=[input_audio, input_video, low_pass_sampling_rate, target_sampling_rate],
        outputs=[input_audio_player, output_audio, plot_output]
    )

demo.queue().launch(share=True)