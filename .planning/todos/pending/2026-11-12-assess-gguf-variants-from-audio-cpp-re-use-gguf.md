---
created: 2026-11-12T00:00:00Z
title: Assess GGUF variants from audio-cpp/RE-USE-GGUF
area: general
severity: minor
files:
  - app.py
  - install.js
  - pinokio.js
  - pinokio.json
  - requirements_full.txt
  - models/
---

## Problem

The user wants to know whether it is realistic to add something from the Hugging Face space
https://huggingface.co/audio-cpp/RE-USE-GGUF (GGUF-quantized model variants of RE-USE) to
improve their own Hugging Face space clone with different variants of the RE-USE model.

The current launcher (this repo) is a Pinokio app for NVIDIA's RE-USE Speech Enhancement
model. It runs full-precision inference via PyTorch + Mamba-SSM (SEMamba architecture),
downloads `nvidia/NVIDIA-NemotronLabs-RE-USE` from Hugging Face at first use, and is
NVIDIA/CUDA-only (Windows + Linux). See `app.py` (model load at line 73), `install.js`
(torch family pins, mamba build), `pinokio.json` (platform/arch/gpu metadata).

Adding GGUF variants would require researching the audio-cpp/RE-USE-GGUF space contents,
GGUF format compatibility with the existing inference pipeline, and whether model variants
can be swapped into the Gradio `enhance()` flow without breaking chunk-wise overlap-add
inference or spectrogram generation.

## Solution

TBD — needs investigation of:
1. What files/variants exist in https://huggingface.co/audio-cpp/RE-USE-GGUF
2. Whether GGUF models can be loaded by the current PyTorch/Mamba stack or need a different
   runtime (e.g., llama.cpp / ggml bindings)
3. How variant selection would be exposed in the Gradio UI (model dropdown in app.py)
4. Any license compatibility between NVIDIA RE-USE and audio-cpp RE-USE-GGUF
