# RE-USE Speech Enhancement - Pinokio Notes

Pinokio launcher for the NVIDIA RE-USE Speech Enhancement model.

[README.md](README.md) covers features, requirements, and troubleshooting.

- Original Space: https://huggingface.co/spaces/nvidia/RE-USE
- Model: https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE

## Launcher files

| File | Purpose |
|------|---------|
| `app.py` | Gradio app; audio-only, refuses to start without CUDA, `share=False` |
| `install.js` | Creates `.env` (Python 3.10), installs dependencies and mamba, then verifies |
| `run.js` | Starts `app.py` and captures the local URL |
| `reset.js` | Removes `.env`, `cache`, `mamba`, legacy `causal-conv1d`; keeps `output/` and `logs/` |
| `update.js` | Updates the launcher and app files |
| `torch.js` | NVIDIA-only PyTorch cu128 installer with fail-fast guard |
| `verify.js` | Runs `python verify_gpu.py` inside `.env` |
| `verify_gpu.py` | GPU/runtime verifier with JSON output |
| `reuse_runtime.py` | Shared mamba selective-scan patch |
| `menu.js` | Dynamic menu, includes **Verify GPU** |
| `pinokio.js` / `pinokio.json` | Pinokio UI and metadata |

## Running with Pinokio

1. Start Pinokio Desktop
2. Add this folder as an app
3. Click **Install** (ends with `python verify_gpu.py`)
4. Click **Run RE-USE** and open the captured local URL
5. Click **Verify GPU** to re-run the check at any time

The launcher uses its own conda environment at `.env`.

## Notes

- The model downloads from Hugging Face on first use
- Mamba is installed as a pure-Python package from tag `v2.3.2.post1`
- `causal-conv1d` is not installed; the optional extension is not needed
- The app uses Mamba's reference selective-scan implementation
- The original space uses `@spaces.GPU` (ZeroGPU); this launcher runs locally
