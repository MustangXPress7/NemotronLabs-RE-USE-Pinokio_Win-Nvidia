# RE-USE Speech Enhancement - Pinokio App

This is a Pinokio launcher for the **NVIDIA RE-USE Speech Enhancement** model.

## Original Space
https://huggingface.co/spaces/nvidia/RE-USE

## Model
https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE

## Features
- Universal speech enhancement for noise, reverberation, clipping, and bandwidth limitation
- Accepts 8 kHz to 48 kHz audio; the STFT is scaled to the input sample rate
- Before/after spectrogram comparison
- Chunk-wise inference to bound model activation memory (3 s chunks, 50% overlap)
- NVIDIA-only install: the launcher refuses to install on machines without an NVIDIA GPU
- One command verifies the CUDA stack and the model runtime (`python verify_gpu.py`)

## Quick Start with Pinokio

1. **Install Pinokio Desktop** from https://pinokio.computer
2. **Add this launcher** to Pinokio (File -> Add App -> Select this folder)
3. **Click "Install"** - creates a conda environment at `.env` with Python 3.10,
   installs FFmpeg, PyTorch cu128, Triton, and a pure-Python `mamba-ssm`, then
   runs `python verify_gpu.py` as the final gate
4. **Click "Run RE-USE"** - launches the Gradio web UI on a local URL
5. **Open Web UI** - click the "Open UI" link when it appears

The first install downloads several GB of PyTorch/CUDA wheels and Python
dependencies. The model itself is downloaded from Hugging Face on first use.

## Supported configurations

| State | Configuration | Notes |
|-------|---------------|-------|
| **Tested** | Windows x64 + NVIDIA GeForce RTX 4070 (12 GB, compute 8.9), torch 2.10.0+cu128 | Install, GPU verification, real inference, and local Gradio launch were executed on this machine |
| **Reasoned** | NVIDIA Quadro RTX 6000 (Turing, compute 7.5) | The installed cu128 wheel lists `sm_75` in its architecture list. This is **inferred from wheel architecture coverage, not measured on that card** |
| **Unsupported** | AMD GPUs, Apple Silicon, CPU-only, Intel Mac | The app hardcodes `device = "cuda"` and the launcher has no fallback. Install fails fast on these machines |
| **Unverified** | Linux x64 + NVIDIA | A Linux install branch exists but was not executed on this machine |

### Requirements

- **OS**: Windows 10/11 or Linux, x64.
- **GPU**: NVIDIA CUDA-capable GPU with a driver that supports CUDA 12.8.
  A separate CUDA Toolkit is not required because the PyTorch cu128 wheels
  bundle the CUDA runtime.
- **Audio formats**: anything `soundfile` can read, such as WAV, FLAC, OGG, and
  MP3. `.m4a` and video containers are **not supported**; convert them to WAV
  first.
- **Disk**: several GB for the PyTorch/CUDA wheels and dependencies.
- **System RAM**: 8 GB or more.

## Manual Installation (without Pinokio)

These are the same steps the launcher uses.

### 1. Create the environment

```bash
conda create -n re-use python=3.10 -y
conda activate re-use
conda install conda-forge::ffmpeg -y
```

### 2. Install PyTorch for CUDA 12.8

```bash
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --force-reinstall --no-deps
```

Windows also needs the Triton runtime used by Mamba:

```bash
uv pip install triton-windows==3.3.1.post19
```

On Linux, install regular Triton instead:

```bash
uv pip install triton
```

### 3. Install Python dependencies

```bash
uv pip install -r requirements_full.txt
```

### 4. Install mamba-ssm as a pure-Python package

No `causal-conv1d` build is required or installed. `reuse_runtime.py` makes the
CUDA selective-scan extension optional and routes the model to Mamba's
reference `selective_scan_ref` implementation.

The pinned tag `v2.3.2.post1` reads `MAMBA_SKIP_CUDA_BUILD`. It does **not** read
the older `MAMBA_KEEP_*` build variable, and `MAMBA_FORCE_BUILD` is unnecessary when
the skip flag is set.

```bash
git clone --branch v2.3.2.post1 --depth 1 https://github.com/state-spaces/mamba.git
cd mamba
MAMBA_SKIP_CUDA_BUILD=TRUE \
  uv pip install --no-build-isolation --no-deps .
cd ..
```

On Windows PowerShell:

```powershell
$env:MAMBA_SKIP_CUDA_BUILD='TRUE'
uv pip install --no-build-isolation --no-deps .
```

`uv` is recommended because Pinokio ships it. Replace `uv pip` with `pip` if you
do not have `uv`.

### 5. Verify and run

```bash
python verify_gpu.py
python app.py
```

## Verification

`verify_gpu.py` prints a JSON report and exits non-zero if anything required is
missing. It checks, in order:

1. `torch` imports and reports its version
2. `torch.cuda.is_available()` is `True`
3. Device name and compute capability
4. The CUDA architecture list contains both `sm_75` and `sm_86`
5. `torchvision` and `torchaudio` import
6. `mamba_ssm` imports after `patch_mamba_selective_scan()`, and
   `selective_scan_ref` is callable
7. `mamba_simple.selective_scan_fn` is bound to `selective_scan_ref`

The JSON keys are: `ok`, `torch`, `torchvision`, `torchaudio`, `cuda_available`,
`device`, `capability`, `arch_list`, `required_arch`, `mamba_ssm`,
`mamba_patch_applied`, `selective_scan_cuda`, `selective_scan_ref`,
`selective_scan_bound`, and `errors`.

In Pinokio, **Install** runs this check automatically and the **Verify GPU**
menu entry re-runs it at any time. `verify.js` is the launcher for that menu
entry.

## What we learned / What we unlearned

**What we learned**

- The installed PyTorch cu128 wheel lists
  `['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`.
  `sm_75` is the Turing target for the Quadro RTX 6000; `sm_86` is the
  binary-compatible path used by the RTX 4070.
- `torchvision==0.15.0` was wrong for torch 2.10.0. The correct matching pin is
  `torchvision==0.25.0`, installed together with `torchaudio==2.10.0`.
- `causal-conv1d` is optional. `mamba_simple` guards the import, and the app uses
  the pure-Python selective scan, so a broken `causal_conv1d_cuda` extension does
  not block inference.
- `soundfile` cannot read `.m4a` or video containers. Convert to WAV first.
- Gradio's public `share=True` tunnel failed with an `h11` error on this machine
  and is not needed for a local launcher, so the app now launches with
  `share=False`.

**What we unlearned**

- The belief that the previous round of edits added GPU support. It did **not**.
  The CUDA 12.8 wheel already covered both `sm_75` and `sm_86`. Those edits fixed
  install robustness and environment correctness instead.
- The claim that the older `MAMBA_KEEP_*` build variable controls the build. The
  pinned `v2.3.2.post1` tag reads `MAMBA_SKIP_CUDA_BUILD`; the other variable is
  simply ignored.
- The claim that the launcher measured VRAM on a Quadro RTX 6000 with a specific
  capacity. No such measurement exists, and that card has 24 GB. The fabricated
  table was removed.
- The claim that video files can be uploaded. Only audio read by `soundfile` is
  supported.

## Memory

No VRAM benchmark was run for this launcher. Do not plan capacity from numbers
that do not exist.

What is known from the code:

- Peak VRAM grows with the input sample rate because the app does not resample
  before building the STFT. The same 3-second chunk costs more at 48 kHz than at
  16 kHz.
- The 3-second chunking bounds model activation memory, not the whole file.
  `app.py` keeps three full-length audio buffers on the GPU (input, enhanced
  output, and overlap-add window sum), so very long recordings add memory in
  proportion to their duration.
- On a small GPU, pre-resample input to 16 kHz and use mono.

## Platform Compatibility

| Platform | Architecture | GPU | Status |
|----------|--------------|-----|--------|
| Windows 10/11 | x64 | NVIDIA RTX 4070 | Tested |
| Windows/Linux | x64 | NVIDIA Quadro RTX 6000 | Reasoned from `sm_75`, not measured |
| Linux | x64 | NVIDIA | Install path exists, not executed here |
| Any | Any | AMD GPUs / Apple Silicon / CPU only | Unsupported; install fails fast |

**Why no Mac/AMD/CPU support?** `pinokio.json` declares
`gpu: ["nvidia"]`, `install.js` and `torch.js` stop with a clear error when no
NVIDIA GPU is detected, and `app.py` uses `device = "cuda"` with no fallback.

## Project Structure

```
RE-USE-pinokio/
├── app.py                 # Main Gradio application
├── install.js             # Pinokio install script (ends with GPU verification)
├── run.js                 # Pinokio start script (captures the local URL)
├── reset.js               # Pinokio reset script (artifacts only)
├── update.js              # Pinokio update script
├── verify.js              # Pinokio "Verify GPU" script
├── verify_gpu.py          # GPU/runtime verifier with JSON output
├── reuse_runtime.py       # Single mamba selective-scan patch
├── pinokio.js             # Pinokio UI generator
├── pinokio.json           # Pinokio metadata
├── menu.js                # Dynamic menu (includes Verify GPU)
├── torch.js               # NVIDIA-only PyTorch installer
├── requirements.txt       # Core dependencies
├── requirements_full.txt  # Full dependencies
├── models/                # Model definitions (Mamba, STFT, Generator)
├── recipes/               # Model configurations (YAML)
├── utils/                 # Utility functions
├── output/                # Enhanced audio output (gitignored)
└── icon.png               # App icon
```

## How It Works

The app uses the **SEMamba** architecture (Selective State Space Model) for
speech enhancement:

1. **Input**: Noisy audio (8-48 kHz) read with `soundfile`
2. **Preprocessing**: STFT scaled to the input sample rate
3. **Enhancement**: SEMamba model processes magnitude and phase spectrograms
4. **Postprocessing**: Overlap-add chunk-wise inference for bounded memory use
5. **Output**: Enhanced audio plus a before/after spectrogram comparison

The Mamba selective scan uses the pure-Python `selective_scan_ref`
implementation because `reuse_runtime.py` makes the CUDA extension import
optional. This keeps installation reproducible across NVIDIA GPUs and avoids
per-machine CUDA extension builds, at the cost of speed compared with compiled
kernels. Triton kernels used elsewhere in Mamba are JIT-compiled at runtime for
the detected GPU.

## Configuration

The model configuration is in
`recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml`.

Key parameters:

- `sampling_rate`: 8000 Hz (internal reference rate; input STFT is scaled from this)
- `n_fft`: 320, `hop_size`: 40, `win_size`: 320
- `compress_factor`: `relu_log1p`
- Chunk size: 3 seconds with 50% overlap

The optional "target sampling rate" and "low-pass sampling rate" fields exist in
the UI, but this local build does not apply resampling or bandwidth extension.
Use an external tool to resample or band-limit input before uploading.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No virtual environment found` during install | An older script ran PyTorch install outside `.env`. Re-run Install; the current `torch.js` receives `conda: '.env'` |
| CUDA out of memory | Pre-resample to 16 kHz, use mono, or split very long recordings. Peak VRAM scales with sample rate and, for long files, duration |
| Mamba import error | Re-run Install. The launcher installs `mamba-ssm` as a pure-Python package; do not build `causal-conv1d` |
| `torch.cuda.is_available() is False` | This launcher has no CPU fallback. Install a recent NVIDIA driver with CUDA 12.8 support, then run **Verify GPU** |
| `LibsndfileError` on `.m4a` or a video file | `soundfile` cannot read `.m4a` or video containers. Convert to WAV first |
| Gradio share link fails | No longer applicable: the app launches with `share=False` and only exposes a local URL |
| Port already in use | Gradio picks the next free port automatically; check the terminal for the captured URL |

## License

The RE-USE model is licensed under NVIDIA's custom license. See the
[original space](https://huggingface.co/spaces/nvidia/RE-USE) for details.

## Credits

- **Model**: NVIDIA Nemotron Labs
- **Architecture**: SEMamba (Mamba-SSM for speech enhancement)
- **Original Space**: https://huggingface.co/spaces/nvidia/RE-USE
- **Pinokio Launcher**: This repository
