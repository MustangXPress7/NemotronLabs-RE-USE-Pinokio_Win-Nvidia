# RE-USE Speech Enhancement - Pinokio App

This is a Pinokio app for the NVIDIA RE-USE Speech Enhancement model from Hugging Face Spaces.

## Original Space
https://huggingface.co/spaces/nvidia/RE-USE

## Structure
- `app.py` - Main Gradio application
- `models/` - Model definitions
- `recipes/` - Model configurations
- `utils/` - Utility functions
- `exp/` - Pre-trained model weights

## Running with Pinokio (Recommended)

1. Start the Pinokio desktop application (this starts the control plane server)
2. Run the app using pterm:

```bash
pterm run "H:\Proyectos Software\Nemotron RE-USE\RE-USE-pinokio"
```

Or if you have the Pinokio CLI in your PATH:
```bash
pterm run "H:\Proyectos Software\Nemotron RE-USE\RE-USE-pinokio" --default "run.js?mode=Default"
```

## Running Directly (Without Pinokio Control Plane)

If you don't want to use the Pinokio control plane, you can run it directly:

### Option 1: Using conda (recommended)

```bash
# Create and activate conda environment
conda create -n re-use python=3.10 -y
conda activate re-use

# Install FFmpeg
conda install conda-forge::ffmpeg -y

# Install Python dependencies
pip install -r requirements_full.txt

# Install mamba-ssm dependencies (pure Python, no CUDA build)
pip install --no-deps setuptools wheel ninja packaging
MAMBA_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=FALSE CAUSAL_CONV1D_FORCE_BUILD=FALSE pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm

# Run the app
python app.py
```

### Option 2: Using the direct runner script

```bash
# Install dependencies
pip install -r requirements_full.txt

# Run the direct runner (handles mamba-ssm installation)
python run_direct.py
```

## Model

The app downloads the RE-USE model from Hugging Face Hub (`nvidia/RE-USE`) on first run.

## Features

- Speech enhancement for diverse degradations
- Support for different sampling rates
- Bandwidth extension capabilities
- Spectrogram visualization
- Audio and video input support

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- FFmpeg (for video input)
- ~8GB VRAM minimum

## Notes

- The model uses Mamba-SSM architecture which requires special installation (handled automatically)
- For long audio files, chunk-wise inference is recommended to avoid OOM errors
- The original space uses `@spaces.GPU` decorator for ZeroGPU - this runs locally on your GPU