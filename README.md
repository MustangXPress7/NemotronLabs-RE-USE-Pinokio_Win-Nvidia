# RE-USE Speech Enhancement - Pinokio App

This is a Pinokio launcher for the **NVIDIA RE-USE Speech Enhancement** model from Hugging Face Spaces.

## Original Space
https://huggingface.co/spaces/nvidia/RE-USE

## Model
https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE

## Features
- Universal speech enhancement for diverse degradations (noise, reverberation, clipping, bandwidth limitation)
- Support for different sampling rates (8kHz - 48kHz)
- Bandwidth extension capabilities
- Real-time spectrogram visualization
- Audio and video input support (.mp4, .mov)
- Chunk-wise inference for long audio files (prevents OOM)

## Quick Start with Pinokio

1. **Install Pinokio Desktop** from https://pinokio.computer
2. **Add this launcher** to Pinokio (File → Add App → Select this folder)
3. **Click "Install"** - Sets up conda environment with Python 3.10, FFmpeg, PyTorch, Mamba-SSM
4. **Click "Run RE-USE"** - Launches the Gradio web UI
5. **Open Web UI** - Click the "Open UI" link when it appears

## Manual Installation (without Pinokio)

```bash
# Create conda environment
conda create -n re-use python=3.10 -y
conda activate re-use

# Install system dependencies
conda install conda-forge::ffmpeg -y

# Install PyTorch (CUDA 12.8)
pip install torch==2.10.0 torchvision==0.15.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Install Mamba-SSM (Windows requires building from source)
# Windows:
git clone https://github.com/Dao-AILab/causal-conv1d.git && cd causal-conv1d && python setup.py install && cd ..
git clone https://github.com/state-spaces/mamba.git && cd mamba && python setup.py install && cd ..

# Linux/Mac:
pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm

# Install Python dependencies
pip install -r requirements_full.txt

# Run the app
python app.py
```

## Requirements
- **Python**: 3.10+
- **GPU**: NVIDIA CUDA-capable (8GB+ VRAM recommended) — **Required**
- **OS**: Windows 10/11, Linux (x64 only)
- **CUDA Toolkit**: 12.8+ (Linux only; Windows uses bundled CUDA in PyTorch wheels)
- **FFmpeg**: Required for video input
- **Disk**: ~5GB for model weights and dependencies

## Platform Compatibility

| Platform | Architecture | GPU | Status |
|----------|-------------|-----|--------|
| Windows 10/11 | x64 | NVIDIA CUDA | ✅ **Tested & Working** |
| Linux | x64 | NVIDIA CUDA | ⚠️ **Should work** (requires CUDA 12.8+ toolkit) |
| Linux | x64 | AMD ROCm | ❌ Not supported (Mamba kernels CUDA-only) |
| Linux | x64 | CPU only | ❌ Too slow (pure Python fallback ~100x slower) |
| macOS | arm64 (Apple Silicon) | Metal/MPS | ❌ Not supported (no CUDA kernels for Mamba-SSM) |
| macOS | x64 (Intel) | — | ❌ Not supported |

**Why not Mac/Linux CPU/AMD?** The SEMamba model uses Mamba-SSM architecture which requires custom CUDA kernels (`selective_scan_cuda`) for the selective scan operation. These kernels:
- Only exist for NVIDIA CUDA
- Have no Metal/MPS implementation for Apple Silicon
- Have no ROCm implementation for AMD
- The pure Python reference implementation (`selective_scan_ref`) is ~10-100x slower and not practical for real-time enhancement

## Project Structure
```
RE-USE-pinokio/
├── app.py                 # Main Gradio application
├── install.js             # Pinokio install script
├── start.js               # Pinokio start script (launches Gradio)
├── reset.js               # Pinokio reset script
├── update.js              # Pinokio update script
├── pinokio.js             # Pinokio UI generator
├── pinokio.json           # Pinokio metadata
├── menu.js                # Dynamic menu
├── torch.js               # Cross-platform PyTorch/Mamba installer
├── requirements.txt       # Core dependencies
├── requirements_full.txt  # Full dependencies
├── models/                # Model definitions (Mamba, STFT, Generator)
├── recipes/               # Model configurations (YAML)
├── utils/                 # Utility functions
├── exp/                   # Pre-trained weights (gitignored)
├── cache/                 # Hugging Face cache (gitignored)
├── logs/                  # Pinokio logs (gitignored)
├── output/                # Enhanced audio output (gitignored)
└── icon.png               # App icon
```

## How It Works

The app uses the **SEMamba** architecture (Selective State Space Model) for speech enhancement:

1. **Input**: Noisy audio (any sample rate 8-48kHz) or video file
2. **Preprocessing**: STFT with configurable n_fft, hop_size, win_size
3. **Enhancement**: SEMamba model processes magnitude + phase spectrograms
4. **Postprocessing**: Overlap-add chunk-wise inference for memory efficiency
5. **Output**: Enhanced audio + side-by-side spectrogram comparison

## Configuration

The model configuration is in `recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml`

Key parameters:
- `sampling_rate`: 16000 Hz (internal processing rate)
- `n_fft`: 320, `hop_size`: 40, `win_size`: 320
- `compress_factor`: 0.5 (magnitude compression)
- Chunk size: 3 seconds with 50% overlap

## Bandwidth Extension

For bandwidth extension, optionally specify:
- **Low-pass sampling rate**: Pre-filter input before enhancement
- **Target sampling rate**: Desired output sample rate

Note: Bandwidth extension performance depends on input characteristics. Apply low-pass filtering beforehand for best results.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM on long audio | App uses chunk-wise inference automatically |
| Mamba import error | Re-run Install (builds causal-conv1d/mamba from source on Windows) |
| No GPU detected | Ensure NVIDIA drivers + CUDA 12.8 installed; app falls back to CPU |
| Port conflict | Pinokio auto-assigns free port via `{{port}}` template |

## License
The RE-USE model is licensed under NVIDIA's custom license. See the [original space](https://huggingface.co/spaces/nvidia/RE-USE) for details.

## Credits
- **Model**: NVIDIA Nemotron Labs
- **Architecture**: SEMamba (Mamba-SSM for speech enhancement)
- **Original Space**: https://huggingface.co/spaces/nvidia/RE-USE
- **Pinokio Launcher**: This repository