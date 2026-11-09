# RE-USE Speech Enhancement - Pinokio App

Pinokio launcher for the **NVIDIA RE-USE Speech Enhancement** model.

- Original Space: https://huggingface.co/spaces/nvidia/RE-USE
- Model: https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE

## Features

- Universal speech enhancement for noise, reverberation, clipping, and bandwidth limitation
- Accepts 8 kHz to 48 kHz audio; the STFT is scaled to the input sample rate
- Before/after spectrogram comparison
- Chunk-wise inference (3 s chunks, 50% overlap) to bound memory use
- NVIDIA-only: the launcher refuses to install without an NVIDIA GPU

## Quick Start with Pinokio

1. Install Pinokio Desktop from https://pinokio.computer
2. Add this folder as an app (File -> Add App -> Select this folder)
3. Click **Install** — creates a conda environment at `.env` (Python 3.10), installs
   FFmpeg, PyTorch cu128, Triton, and a pure-Python `mamba-ssm`, then verifies the GPU
4. Click **Run RE-USE** and open the captured local URL
5. Use **Verify GPU** at any time to re-check the CUDA stack

The model downloads from Hugging Face on first use.

## Requirements

- Windows 10/11 or Linux, x64
- NVIDIA GPU with a driver that supports CUDA 12.8 (no separate CUDA Toolkit needed)
- Several GB of disk for PyTorch/CUDA wheels and dependencies
- Audio readable by `soundfile` (WAV, FLAC, OGG, MP3). `.m4a` and video containers
  are not supported — convert to WAV first

## Verification

Install ends with `python verify_gpu.py`, which prints a JSON report and exits
non-zero on failure. The **Verify GPU** menu entry re-runs it.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Pre-resample to 16 kHz, use mono, or split long recordings |
| `torch.cuda.is_available() is False` | Install a recent NVIDIA driver with CUDA 12.8 support, then run **Verify GPU** |
| `LibsndfileError` on `.m4a` or video | Convert the file to WAV first |
| Mamba import error | Re-run **Install**; do not build `causal-conv1d` |

## License

The RE-USE model is licensed under NVIDIA's custom license. See the
[original space](https://huggingface.co/spaces/nvidia/RE-USE) for details.

## Credits

- **Model**: NVIDIA Nemotron Labs
- **Architecture**: SEMamba (Mamba-SSM for speech enhancement)
- **Original Space**: https://huggingface.co/spaces/nvidia/RE-USE
- **Pinokio Launcher**: this repository
