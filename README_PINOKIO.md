# RE-USE Speech Enhancement - Pinokio Notes

This is a Pinokio app for the NVIDIA RE-USE Speech Enhancement model.

[README.md](README.md) is the single source of truth for installation, supported
configurations, verification, memory notes, and troubleshooting. This file only
covers the launcher itself.

## Original Space
https://huggingface.co/spaces/nvidia/RE-USE

## Model
https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE

## Support in one line

- **Tested**: Windows x64 + NVIDIA RTX 4070 (12 GB), torch 2.10.0+cu128.
- **Reasoned**: NVIDIA Quadro RTX 6000 via the wheel's `sm_75` coverage, not measured.
- **Unsupported**: AMD GPUs, Apple Silicon, CPU-only. Install fails fast.

## Launcher files

| File | Purpose |
|------|---------|
| `app.py` | Gradio app; audio-only, refuses to start without CUDA, `share=False` |
| `install.js` | Creates `.env` (Python 3.10), installs FFmpeg, torch cu128, deps, mamba, then verifies |
| `run.js` | Starts `app.py` and captures the local URL for the menu |
| `reset.js` | Removes `.env`, `cache`, `mamba`, and legacy `causal-conv1d`; keeps `output/` and `logs/` |
| `update.js` | Updates the launcher and app files |
| `torch.js` | NVIDIA-only cu128 PyTorch install with a fail-fast guard |
| `verify.js` | Runs `python verify_gpu.py` inside `.env` |
| `verify_gpu.py` | JSON GPU/runtime verifier; exits non-zero on failure |
| `reuse_runtime.py` | One shared mamba selective-scan patch used by app and verifier |
| `menu.js` | Dynamic menu; includes **Verify GPU** |
| `pinokio.js` | Pinokio UI generator |
| `pinokio.json` | Metadata: `win32`/`linux`, `x64`, `nvidia` |

## Running with Pinokio

1. Start Pinokio Desktop.
2. Add this folder as an app.
3. Click **Install**. It ends with `python verify_gpu.py`.
4. Click **Run RE-USE** and open the captured local URL.
5. Click **Verify GPU** at any time to re-run `verify_gpu.py`.

The launcher creates its conda environment at `.env` and does not use the system
conda environment.

## Notes

- The model downloads from Hugging Face on first use.
- Mamba is installed as a pure-Python package from tag `v2.3.2.post1` with
  `MAMBA_SKIP_CUDA_BUILD=TRUE`. That tag does not read the older `MAMBA_KEEP_*`
  build variable.
- `causal-conv1d` is not installed; the broken optional extension is not needed.
- The app uses Mamba's reference selective-scan implementation, so there is no
  per-GPU CUDA extension build (Triton kernels are JIT-compiled at runtime).
- The original space uses `@spaces.GPU` for ZeroGPU; this launcher runs locally
  on your NVIDIA GPU.
- No VRAM benchmark is published for this launcher. See [README.md](README.md)
  for what is and is not known about memory use.
