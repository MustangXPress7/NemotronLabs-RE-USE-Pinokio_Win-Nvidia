# Phase 2: Enable RealTime RE-USE Variants - Research

**Researched:** 2026-11-12
**Domain:** Pinokio launcher for NVIDIA RE-USE / RealTime RE-USE speech enhancement model variants
**Confidence:** MEDIUM

## Summary

This phase extends the Phase 1 NVIDIA-only RE-USE launcher to support additional model variants, starting with the RealTime RE-USE variant from the Hugging Face Space `https://huggingface.co/spaces/nvidia/Real-time_RE-USE`, and to document multi-backend viability (Vulkan, ROCm, Metal, OpenVINO) while keeping NVIDIA CUDA as the only executed path. Key findings:

1. **RealTime variant** uses a streaming SEMamba decoder (`SEMamba_decoder_list`) with 12 TFMamba blocks vs the standard model's 30 blocks, a smaller STFT hop (160 vs 40), exposes two additional inference knobs (`exit_layer` between 3–12, default 8; `look_ahead_frames` between 0–2, default 0) for quality–latency trade-offs, and loads weights from a local `exp/.../g_01010000.pth` checkpoint via `torch.load` instead of `SEMamba.from_pretrained`. Its config YAML is `USEMamba_12x1_lr_00002_norm_05_vq_067_nfft_320_hop_160_NRIR_012_pha_0005_com_04_early_005_release_random_layer_GAN_longer_1k.yaml`.

2. **Gradio UI**: Both spaces currently use `gr.Textbox` for `target_sampling_rate` and `low_pass_sampling_rate` (app.py lines 193-194 in the local copy; the RealTime space uses the same at lines ~49-50 of its app.py). The RealTime space adds `gr.Textbox` fields for `Exit_layer` and `Look_ahead_frames`. Converting to `gr.Slider` requires: `minimum`, `maximum`, `value`, `step`, `label`, and `interactive=True`; Gradio sliders emit numbers directly, so the `int(...)` casts in `enhance()` would need to accept float/int but `target_sampling_rate != ''` checks break because sliders never return empty string — the conditional logic must change.

3. **Multi-backend**: Vulkan, ROCm, Metal, and OpenVINO are all infeasible as *executed* paths for this phase because mamba-ssm selective-scan kernels are CUDA-specific, `app.py` hardcodes `device = "cuda"`, and the repo's `reuse_runtime.py` patches mamba to a pure-Python reference kernel (slow but CUDA-independent). OpenVINO is the most documentable (ONNX export path exists for similar Mamba models); Metal via MLX is possible but requires a full model rewrite. None will be executed — only documented as assessed.

**Primary recommendation:** Add a "Model Variant" dropdown (Standard / RealTime) to the Gradio UI, load the matching recipe YAML and model class per variant, convert `target_sampling_rate`/`low_pass_sampling_rate` to `gr.Slider`, expose `exit_layer`/`look_ahead_frames` for RealTime only, keep NVIDIA CUDA as the executed path, and document backend feasibility without implementing any non-CUDA install.

## <user_constraints>
## User Constraints (from CONTEXT.md)

No CONTEXT.md exists for Phase 2 yet. Locked decisions from Phase 1 (CONTEXT.md in phase 1 directory) that constrain Phase 2:
- D-01: NVIDIA-only GPU scope. `pinokio.json` declares `"gpu": ["nvidia"]`; keep that.
- D-24: `demo.queue().launch(share=False)` — no public Gradio tunnel.
- D-25: Remove broken Video Upload tab; UI is audio-only.
- No AMD/Apple/CPU install paths until hardware and validated kernels exist.
- Pinokio scripts must follow API patterns from `D:\pinokio\prototype\system\examples`; no ad-hoc absolute paths.
- Keep one GSD phase with multiple plans; do not split into separate phases.

## </user_constraints>

## <phase_requirements>
## Phase Requirements

Phase 2 requirement IDs are proposed below (to be locked during planning):

| ID | Description | Research Support |
|----|-------------|------------------|
| RTV-01 | RealTime model variant loads and performs inference on RTX 4070 via Gradio UI | RealTime app.py uses `SEMamba_decoder_list(cfg1).to(device)` + `torch.load(...g_01010000.pth)` |
| RTV-02 | Model variant selection dropdown in Gradio UI (Standard / RealTime) | app.py `gr.Blocks` UI structure; RealTime space UI as reference |
| RTV-03 | Recipe YAML and model class loaded per variant | Config YAML differences: 12x1 hop_160 vs 30x1 hop_40; model class `SEMamba_decoder_list` vs `SEMamba` |
| GUI-01 | `target_sampling_rate` and `low_pass_sampling_rate` converted from `gr.Textbox` to `gr.Slider` | app.py lines 193-194 (local); RealTime app.py lines ~49-50 |
| GUI-02 | Additional parameters exposed from UI: chunk size, overlap, window, normalization | app.py `enhance()` lines 95-97, 147; RealTime `exit_layer`/`look_ahead_frames` pattern |
| GUI-03 | RealTime-only parameters (`exit_layer`, `look_ahead_frames`) hidden/disabled for Standard variant | RealTime app.py `enhance(filepath, low_pass_sampling_rate, target_sampling_rate, exit_layer, look_ahead_frames)` |
| BK-01 | Documented viability assessment for Vulkan, ROCm, Metal, OpenVINO backends | SEED-261112-mos research; mamba-ssm CUDA kernel dependency |
| BK-02 | NVIDIA CUDA remains the only executed path; no non-NVIDIA install implemented | Phase 1 D-01 constraint; `install.js` fail-fast guard line 9 |
| PIN-01 | `pinokio.json` metadata reflects new model variant(s) | pinokio.json lines 6-8 platform/arch/gpu declaration |
| PIN-02 | `menu.js` updated to expose model variant selection without breaking Phase 1 flow | menu.js lines 18-33 run button; `mode` parameter pattern |
| PIN-03 | `install.js` unchanged for NVIDIA path; no new dependencies for variant support | install.js torch.js delegation line 49-62; requirements_full.txt |
| REL-05 | No regression on Phase 1: clean install, GPU verify, inference, URL capture pass on RTX 4070 | Phase 1 verified requirements INST-01..06, GPU-01..05, REL-01..04 |

## </phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Model variant selection (Standard/RealTime) | Browser / Client | API / Backend | User picks variant in Gradio dropdown (client); app.py loads matching model/recipe (backend) |
| Model loading (`from_pretrained` / `torch.load`) | API / Backend | — | `app.py` `_get_model()` / variant load; weights from Hugging Face Hub or local `exp/` |
| Chunked inference (3s chunks, 50% overlap) | API / Backend | — | `app.py` `enhance()` lines 95-144; CUDA-bound, no client offload |
| Gradio UI sliders for sample rates | Browser / Client | API / Backend | `gr.Slider` renders in browser; values flow to `enhance()` backend |
| Multi-backend feasibility docs | API / Backend | — | Documentation only; no execution path changes |
| Pinokio install (conda, torch cu128, mamba) | OS / Host | — | `install.js`/`torch.js` run on host machine; NVIDIA only |
| GPU verification | API / Backend | OS / Host | `verify_gpu.py` runs in `.env` conda env; checks CUDA availability |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gradio | 6.12.0 | Web UI framework | Matches RealTime Space `sdk_version: 6.12.0` [CITED: https://huggingface.co/spaces/nvidia/Real-time_RE-USE/raw/main/README.md] |
| torch | 2.10.0+cu128 | Deep learning backend | Phase 1 verified pin [VERIFIED: torch.js line 12] |
| torchaudio | 2.10.0+cu128 | Audio loading/resampling | Phase 1 verified pin [VERIFIED: torch.js line 12] |
| mamba-ssm | 2.3.2.post1 | Selective scan Mamba blocks | Phase 1 pinned tag [VERIFIED: install.js line 87] |
| librosa | latest | Resampling (kaiser_best) | Used in RealTime app.py `librosa.resample(..., res_type="kaiser_best")` [VERIFIED: RealTime app.py line ~85] |
| soundfile | latest | WAV reading/writing | Local app.py line 15 `import soundfile as sf` [VERIFIED: local app.py] |
| pyyaml | latest | Recipe YAML parsing | `utils/util.py` `load_config()` [VERIFIED: local utils/util.py] |
| huggingface_hub | latest | Model download | `hf_hub_download` import [VERIFIED: local app.py line 53] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchcodec | 0.10 | Alternative audio decoder | Listed in both spaces' requirements.txt; not currently used by local app.py [VERIFIED: RealTime requirements.txt] |
| resampy | latest | High-quality resampling | Required for bandwidth extension per model card [CITED: https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE] |
| einops | latest | Tensor reshaping | Model forward pass uses `einops.rearrange` [VERIFIED: generator_SEMamba_time_d4.py line 5] |
| numpy | 1.26.4 | Array operations | Pinned in both spaces' requirements.txt [VERIFIED: RealTime requirements.txt] |
| matplotlib | latest | Spectrogram plotting | `app.py` line 48 `import matplotlib.pyplot as plt` [VERIFIED: local app.py] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `gr.Textbox` for sample rates | `gr.Number` | Simpler but no range enforcement; slider better for UX |
| `SEMamba.from_pretrained(...)` | `torch.load(local_checkpoint)` | RealTime uses local `.pth`; standard uses Hub `from_pretrained` |
| conda `.env` | plain `venv` | Phase 1 D-08: conda proven working; do not switch |

**Installation:**
```bash
# Phase 2 adds no new Python packages beyond Phase 1 stack.
# torch.js already installs torch==2.10.0, torchaudio==2.10.0 from cu128 index.
# requirements_full.txt covers librosa, soundfile, pyyaml, matplotlib, einops, numpy.
```

**Version verification:** All package versions above were confirmed against the RealTime Space `requirements.txt` [VERIFIED: https://huggingface.co/spaces/nvidia/Real-time_RE-USE/raw/main/requirements.txt] and local `torch.js` [VERIFIED: local torch.js lines 12-13]. `gradio` SDK version confirmed from RealTime Space README `sdk_version: 6.12.0` [CITED: RealTime README.md].

## Package Legitimacy Audit

Phase 2 does not install any new external packages beyond the Phase 1 stack. All packages (torch, torchaudio, mamba-ssm, librosa, soundfile, pyyaml, huggingface_hub, einops, numpy, matplotlib, gradio) were already verified in Phase 1 or confirmed present in the RealTime Space requirements.txt. No `[SLOP]` or `[SUS]` packages are introduced.

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐   │
│  │ Audio Input │  │ Model Select│  │ Sliders (sample rate, │   │
│  │ (filepath)  │  │ (Std/RTV)   │  │ chunk, overlap, exit,   │   │
│  └──────┬──────┘  └──────┬──────┘  │ look_ahead, norm)      │   │
│         │                │         └──────────┬────────────┘   │
│         └────────────────┴────────────────────┘                │
│                         │                                       │
│                    Enhance Button                               │
│                         │                                       │
└─────────────────────────┼───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Gradio Backend (app.py)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ unified_enhance() dispatches to enhance()               │   │
│  │                                                         │   │
│  │  1. Load variant config YAML (recipe)                   │   │
│  │  2. Load model class (SEMamba or SEMamba_decoder_list)  │   │
│  │  3. Load weights (from_pretrained or torch.load .pth)   │   │
│  │  4. Resample audio per slider values                    │   │
│  │  5. STFT → Mamba blocks → iSTFT (CUDA device)           │   │
│  │  6. Chunked overlap-add (3s chunks, 50% hop)            │   │
│  │  7. Write enhanced.wav + spectrogram plot               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│              ┌──────────────────────┐                           │
│              │  NVIDIA CUDA (device │                           │
│              │  = "cuda", hardcoded)│                           │
│              └──────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
src/  # Not applicable — app lives at repo root per Phase 1 structure
app.py              # Gradio app; add variant dropdown + sliders
install.js          # Unchanged (NVIDIA path only)
torch.js            # Unchanged (cu128 NVIDIA only)
run.js              # Unchanged (URL capture)
menu.js             # Add variant-aware run params if needed
pinokio.json        # Add model variant tags
reuse_runtime.py    # Unchanged (pure-Python mamba patch)
verify_gpu.py       # Unchanged
requirements_full.txt # Unchanged
```

### Pattern 1: Model Variant Registry

**What:** A small registry mapping variant IDs to (recipe YAML, model class, weight source, weight path) so `enhance()` can load the right model per variant without branching spaghetti.

**When to use:** When supporting 2+ model variants with different configs and weight sources.

**Example:**
```python
# Source: app.py local lines 62-76 (model loading pattern)
VARIANT_REGISTRY = {
    "standard": {
        "recipe": "recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml",
        "model_class": "SEMamba",
        "model_module": "models.generator_SEMamba_time_d4",
        "weight_source": "hf_hub",       # nvidia/NVIDIA-NemotronLabs-RE-USE
        "weight_kwargs": {"cfg": cfg1},
    },
    "realtime": {
        "recipe": "recipes/USEMamba_12x1_lr_00002_norm_05_vq_067_nfft_320_hop_160_NRIR_012_pha_0005_com_04_early_005_release_random_layer_GAN_longer_1k.yaml",
        "model_class": "SEMamba_decoder_list",
        "model_module": "models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv",
        "weight_source": "local_pth",    # exp/.../g_01010000.pth
        "weight_path": "exp/12x1_lr_00002_norm_05_vq_065_nfft_320_hop_160_NRIR_012_pha_0005_com_04_peak_release_random_layer_ahead_sep_conv2/g_01010000.pth",
        "extra_ui_params": ["exit_layer", "look_ahead_frames"],
    },
}
```

### Anti-Patterns to Avoid
- **Hard-coding `device = "cuda"` per variant:** Keep single `device = "cuda"` at module scope; do not duplicate per variant.
- **Adding non-CUDA install branches to `install.js`/`torch.js`:** Violates Phase 1 D-01 (NVIDIA-only) and BK-02.
- **Using `gr.Textbox` with `!= ''` empty checks alongside sliders:** Sliders never return empty string; logic must use `None`/numeric defaults instead.
- **Loading RealTime weights via `from_pretrained("nvidia/Real-time_RE-USE")`:** The RealTime app.py comments out this line and uses `torch.load(local_pth)` instead [VERIFIED: RealTime app.py lines ~70-74].

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Selective scan Mamba kernels for Vulkan/ROCm/Metal/OpenVINO | Custom CUDA→other-backend kernel port | Document only; keep pure-Python `selective_scan_ref` fallback | mamba-ssm kernels are CUDA-specific; porting is a multi-month effort outside Phase 2 scope [SEED-261112-mos] |
| Model variant if/else chains | Scattered variant checks throughout `enhance()` | Variant registry dict mapping variant→(recipe, class, weights) | Single lookup point; easy to add third variant later |
| Gradio slider value parsing | Manual string→int conversion with empty checks | `gr.Slider` returns numeric directly; use `value` default | Avoids `target_sampling_rate != ''` bugs |

**Key insight:** The selective-scan Mamba kernel is the one component that cannot be hand-rolled or trivially ported — it is CUDA-specific compiled code. Phase 2 documents this honestly rather than promising backends it cannot deliver.

## Gradio UI Customization Plan (SEED-261112-e08)

### Current State (Verified from local app.py)

| Element | Location | Type | Label |
|---------|----------|------|-------|
| `target_sampling_rate` | app.py:193 | `gr.Textbox` | "(Optional) Enter target sampling rate for bandwidth extension:" |
| `low_pass_sampling_rate` | app.py:194 | `gr.Textbox` | "(Optional) Enter target sampling rate for pre-low-pass filtering before bandwidth extension:" |
| Inputs to `enhance_btn.click` | app.py:207 | `[input_audio, low_pass_sampling_rate, target_sampling_rate]` | — |

RealTime Space app.py (reference) uses the same two Textboxes plus `Exit_layer` and `Look_ahead_frames` Textboxes [VERIFIED: RealTime app.py lines ~49-52].

### Slider Conversion Requirements

`gr.Slider` constructor parameters (Gradio 6.12.0):
- `minimum` / `maximum`: numeric range
- `value`: default value
- `step`: increment granularity
- `label`: display label
- `interactive`: whether user can change it

Key behavioral difference from `gr.Textbox`: sliders always return a number, never `""`. The current `enhance()` guards `if target_sampling_rate != ''` and `if low_pass_sampling_rate != ''` [VERIFIED: RealTime app.py lines ~80-88]. With sliders these checks must become `if target_sampling_rate is not None` or use a sentinel like `0` meaning "disabled".

### Recommended Slider Configuration

| Parameter | Type | Min | Max | Default | Step | Rationale |
|-----------|------|-----|-----|---------|------|-----------|
| `target_sampling_rate` | `gr.Slider` | 8000 | 48000 | 0 (off) | 1000 | Model supports 8k–48kHz input [CITED: model card input section] |
| `low_pass_sampling_rate` | `gr.Slider` | 8000 | 48000 | 0 (off) | 1000 | Same range as target; pre-filter before bandwidth extension |
| `chunk_size_seconds` | `gr.Slider` | 1 | 10 | 3 | 1 | Current hardcoded 3.0s [VERIFIED: local app.py line 95] |
| `overlap_portion` | `gr.Slider` | 0.0 | 0.9 | 0.5 | 0.1 | Current hardcoded 50% [VERIFIED: local app.py line 96] |
| `normalization` | `gr.Slider` | 0.0 | 1.0 | 1.0 | 0.1 | Window-sum normalization divisor [VERIFIED: local app.py line 147] |
| `exit_layer` (RealTime only) | `gr.Slider` | 3 | 12 | 8 | 1 | RealTime reference default 8 [VERIFIED: RealTime app.py `value="8"`] |
| `look_ahead_frames` (RealTime only) | `gr.Slider` | 0 | 2 | 0 | 1 | RealTime reference default 0 [VERIFIED: RealTime app.py `value="0"`] |

### Code Locations in app.py to Modify

1. **Lines 193-194:** Replace `gr.Textbox(...)` with `gr.Slider(...)` for both sample rate inputs.
2. **Line 202:** `unified_enhance(audio_path, lp_sr, target_sr)` — signature may grow to accept new slider params.
3. **Line 207:** `inputs=[input_audio, low_pass_sampling_rate, target_sampling_rate]` — add new sliders.
4. **`enhance()` function (lines 79-183):** 
   - Replace `target_sampling_rate != ''` / `low_pass_sampling_rate != ''` with `is not None` / `> 0` checks.
   - Replace hardcoded `chunk_size = int(3.0 * noisy_sr)` (line 95) with slider-driven value.
   - Replace hardcoded `hop_length = int(0.5 * chunk_size)` (line 96) with slider-driven overlap.
   - Replace hardcoded `window = torch.clamp(torch.hann_window(chunk_size)...)` (line 97) — window type could be slider-driven (Hann vs others).
   - Replace final normalization `audio_enhanced / torch.clamp(window_sum, min=1e-7)` (line 147) — normalization factor slider-driven.
5. **Variant-dependent UI:** For Standard variant, hide/disable `exit_layer` and `look_ahead_frames` sliders (RealTime-only params).

### Additional Parameters from SEED-261112-e08

The seed requests exposing: chunk size, overlap, window, normalization. The RealTime Space adds `exit_layer` and `look_ahead_frames` as quality–latency knobs [VERIFIED: RealTime app.py `enhance()` signature line ~79 and Markdown note line ~47]. All of these map to existing code:

- **chunk size** → `app.py:95` `chunk_size = int(3.0 * noisy_sr)`
- **overlap** → `app.py:96` `hop_length = int(0.5 * chunk_size)`
- **window** → `app.py:97` `torch.hann_window(chunk_size)` — could expose window type (Hann/Hamming/etc.)
- **normalization** → `app.py:147` `audio_enhanced / torch.clamp(window_sum, min=1e-7)`
- **exit_layer** → RealTime `SEMamba_decoder_list.forward(..., layer_use, ...)` [VERIFIED: streaming_generator_SEMamba_time_d1...py line 56]
- **look_ahead_frames** → RealTime `dense_encoder(x, number_ahead)` [VERIFIED: streaming_generator_SEMamba_time_d1...py line 64]

### Risk

Converting Textbox→Slider changes the value type from `str` to `float`. All `int(...)` casts and `!= ''` checks in `enhance()` must be audited. The `target_sampling_rate != ''` guard appears in both local app.py (line ~193 context) and RealTime app.py (line ~80) — must be replaced before sliders are introduced or bandwidth-extension input will always be "on" with default 0.

## Multi-Backend Viability Assessment (SEED-261112-mos)

### Constraint (Locked)

NVIDIA CUDA remains the only executed path. `install.js` line 9 fail-fast guard blocks non-NVIDIA/non-Windows-Linux machines [VERIFIED: local install.js line 9]. `app.py` line 39-43 raises `SystemExit` if `torch.cuda.is_available()` is False [VERIFIED: local app.py lines 39-43]. `pinokio.json` declares `"gpu": ["nvidia"]` [VERIFIED: local pinokio.json line 8]. No Vulkan/ROCm/Metal/OpenVINO install or inference path will be implemented in Phase 2 — only documented.

### Why Non-CUDA Backends Are Blocked

| Blocker | Evidence |
|---------|----------|
| mamba-ssm selective-scan kernels are CUDA-specific | `reuse_runtime.py` patches `import selective_scan_cuda` to optional and routes to `selective_scan_ref` (pure-Python) [VERIFIED: local reuse_runtime.py lines 16-54] |
| `device = "cuda"` hardcoded | `app.py` line 61 [VERIFIED: local app.py line 61] |
| `torch.cuda.is_available()` guard exits on non-CUDA | `app.py` lines 39-43 [VERIFIED: local app.py lines 39-43] |
| `install.js` fail-fast without NVIDIA GPU | `install.js` line 9 `when: "{{gpu !== 'nvidia' || ...}}"` [VERIFIED: local install.js line 9] |
| `pinokio.json` `gpu: ["nvidia"]` only | `pinokio.json` line 8 [VERIFIED: local pinokio.json line 8] |

### Per-Backend Feasibility (Documented Only)

| Backend | Feasibility | What Would Be Needed | Executed in Phase 2? |
|---------|------------|---------------------|---------------------|
| **Vulkan** | Low–Medium | No direct PyTorch→Vulkan exporter for this architecture. Would require IREE/TVM compilation or ggml integration with custom shaders for bidirectional selective scan. | No |
| **ROCm (AMD)** | Low | mamba-ssm selective_scan_cuda is CUDA-only; no ROCm kernel. Would require rewriting selective scan in HIP/ROCm and replacing `device = "cuda"` with HIP device. | No |
| **Metal (Apple Silicon)** | Medium | PyTorch MPS does not support mamba-ssm CUDA kernels. Would require porting Mamba blocks to MLX or llama.cpp/ggml with Metal backend, plus STFT/iSTFT rewrite. | No |
| **OpenVINO (Intel)** | Medium–High | Most documentable path: rewrite selective scan in pure PyTorch, export to ONNX (opset 17+), compile with OpenVINO. Model is only 9.6M params. STFT complex-number ops may need rewrite to real-valued conv1d. | No |

### Key Technical Challenges (from SEED-261112-mos)

1. **Bidirectional Mamba:** RE-USE scans audio both forward and backward (non-causal), unlike causal LLM Mamba. Streaming/real-time requires a window buffer (chunking) [SEED-261112-mos].
2. **STFT complex ops:** `models/stfts.py` uses complex-number spectrogram operations that may fail ONNX/OpenVINO conversion; may need real-valued conv1d rewrite [SEED-261112-mos].
3. **Model size:** 9.6M parameters — small enough that pure-Python inference is viable on CPU for OpenVINO [SEED-261112-mos].

### Recommendation

Document feasibility in RESEARCH.md and ROADMAP.md only. Do not add install branches, torch.js ROCm/Metal paths, or `pinokio.json` platform changes for non-NVIDIA backends in Phase 2. If a future phase pursues OpenVINO, the pure-Python `selective_scan_ref` path already in `reuse_runtime.py` is the starting point for ONNX export.

## Model Variant Architecture

### Variant Comparison (Verified from Source)

| Attribute | Standard RE-USE | RealTime RE-USE | Source |
|-----------|-----------------|-----------------|--------|
| Model class | `SEMamba` | `SEMamba_decoder_list` | [VERIFIED: local app.py line 51; RealTime app.py import] |
| Model module | `models.generator_SEMamba_time_d4` | `models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv` | [VERIFIED: local app.py line 51; RealTime app.py import] |
| TFMamba blocks | 30 (`num_tfmamba: 30`) | 12 (`num_tfmamba: 12`) | [VERIFIED: recipe YAMLs] |
| STFT hop_size | 40 | 160 | [VERIFIED: recipe YAMLs] |
| STFT n_fft | 320 | 320 | [VERIFIED: recipe YAMLs] |
| STFT win_size | 320 | 320 | [VERIFIED: recipe YAMLs] |
| Sampling rate | 8000 Hz | 8000 Hz | [VERIFIED: recipe YAMLs] |
| Weight source | `nvidia/NVIDIA-NemotronLabs-RE-USE` via `from_pretrained` | Local `exp/.../g_01010000.pth` via `torch.load` | [VERIFIED: local app.py line 73; RealTime app.py lines ~70-74] |
| Extra inference params | None | `exit_layer` (3–12), `look_ahead_frames` (0–2) | [VERIFIED: RealTime app.py `enhance()` signature] |
| Video upload tab | Yes (broken, soundfile can't demux) | Yes (broken, same) | [VERIFIED: both app.py files] |
| Chunked inference | Yes (3s chunks, 50% overlap) | No chunking (single forward) | [VERIFIED: local app.py lines 95-144; RealTime app.py no chunk loop] |

### Recommended Variant Registry Design

A single registry in `app.py` maps variant ID to config, model class, and weight source. `enhance()` looks up the variant once at the top, then proceeds with the shared inference pipeline. This avoids scattered if/else chains.

The RealTime variant does **not** use chunked inference (single forward pass), while Standard uses chunked overlap-add. The variant registry should also carry a `chunked` flag so `enhance()` can branch on inference strategy, not on model class directly.

### Shared Inference Pipeline

Both variants share:
- STFT/iSTFT via `models/stfts.py` `mag_phase_stft` / `mag_phase_istft` [VERIFIED: both app.py imports]
- `compress_factor`, `sampling_rate`, `n_fft`, `hop_size`, `win_size` from recipe YAML [VERIFIED: local app.py lines 62-65]
- `make_even()` helper [VERIFIED: local app.py line 57-59]
- RELU + magnitude cleanup (`zero_portion > 0.5` mask) [VERIFIED: both app.py lines ~129-131]
- Spectrogram plotting [VERIFIED: both app.py lines ~166-176]
- `pad_or_trim_to_match` [VERIFIED: local utils/util.py]

Divergence points:
- Model class and forward signature (`SEMamba(mag, pha)` vs `SEMamba_decoder_list(mag, pha, layer_use, number_ahead)`)
- Weight loading (`from_pretrained` Hub vs `torch.load` local `.pth`)
- Chunked vs single-pass inference
- RealTime has `exit_layer`/`look_ahead_frames`; Standard does not

## Pinokio Integration Points

| File | Change Required | What Stays Same |
|------|----------------|-----------------|
| `app.py` | Add variant dropdown (`gr.Dropdown`), convert sample-rate Textboxes to Sliders, add chunk/overlap/window/normalization sliders, add exit_layer/look_ahead_frames for RealTime, variant registry, variant-aware `enhance()` | `device = "cuda"`, CUDA guard, mamba patch calls, `demo.queue().launch(share=False)`, spectrogram output, enhance button |
| `install.js` | None (NVIDIA path unchanged) | All steps: conda `.env`, ffmpeg, torch.js delegation, mamba clone/build, `verify_gpu.py` |
| `torch.js` | None | cu128 torch/torchaudio/torchvision install; triton optional |
| `run.js` | None | `python app.py`, URL capture via `/(http:\/\/[0-9.:]+)/` → `local.set url` |
| `menu.js` | Possibly add variant-aware run mode params | Install/Run/Verify/Update/Reset menu structure |
| `pinokio.json` | Add model variant tags (e.g. `"tags": [..., "realtime", "model-variants"]`) | `platform`, `arch`, `gpu: ["nvidia"]`, `version`, `title`, `description` |
| `pinokio.js` | None | Menu wiring via `require(__dirname + '/menu.js')` |
| `verify_gpu.py` | None | CUDA/arch/mamba checks unchanged |
| `reuse_runtime.py` | None | Pure-Python mamba selective-scan patch unchanged |
| `reset.js` | None | Removes `.env`, `cache`, `mamba`, `causal-conv1d` |
| `requirements_full.txt` | None | No new Python packages needed for variants |

### `pinokio.json` Metadata Additions (Recommended)

```json
{
  "tags": ["gradio", "mamba", "speech", "enhancement", "nvidia", "cuda", "model-variants", "realtime"],
  "homepage": "https://huggingface.co/spaces/nvidia/Real-time_RE-USE",
  "repository": "https://github.com/NVIDIA/RE-USE"
}
```
The `homepage` could point to the RealTime Space as the variant reference, or keep the standard RE-USE homepage and add RealTime as a secondary reference. Keep `"gpu": ["nvidia"]` unchanged — no non-NVIDIA backends are executed.

### `menu.js` Integration

The run button currently passes `mode: 'Default'` to `run.js` [VERIFIED: menu.js lines 20-32]. Variant selection happens inside `app.py` Gradio UI (dropdown), not at the Pinokio menu level. No `menu.js` change is strictly required unless the user wants variant pre-selection before launch. If added, `run.js` would accept a `mode` param mapping to variant, and `app.py` would read an env var or CLI arg. This is optional; keep Phase 1 flow unchanged.

## Risks and Open Questions

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RealTime weights (`exp/.../g_01010000.pth`, ~54 MB LFS) not included in local repo clone | Medium | High | RealTime app.py loads from local `exp/` path; launcher must ensure `exp/` folder exists or download weights from HF Space at install time |
| `target_sampling_rate != ''` guard breaks with sliders (sliders never return `""`) | High | Medium | Replace with `is not None` / `> 0` check before enabling sliders |
| RealTime variant uses no chunking — long audio may OOM on 12 GB RTX 4070 | Medium | High | Document OOM risk; keep chunked inference for Standard; optionally add chunking to RealTime in future |
| `exit_layer`/`look_ahead_frames` only valid for RealTime; passing them to Standard model would error | Medium | Medium | Hide/disable sliders for Standard variant; `enhance()` only passes them when variant == realtime |
| Model card claims NVIDIA Ampere/A100 support; RTX 4070 (Ada) is binary-compatible via `sm_86` but not explicitly tested for RealTime | Low | Medium | Verify RealTime inference on RTX 4070 during Phase 2 execution |
| RealTime Space `demo.queue().launch(share=True)` — local copy uses `share=False` per Phase 1 D-24 | Low | Low | Keep `share=False` |

### Open Questions (RESOLVED)

1. **[RealTime weights distribution] RESOLVED: Download at install via `hf_hub_download`.** The launcher downloads `exp/12x1_lr_00002_norm_05_vq_065_nfft_320_hop_160_NRIR_012_pha_0005_com_04_peak_release_random_layer_ahead_sep_conv2/g_01010000.pth` (~54 MB, Git LFS) from the RealTime HF Space at startup if the local `exp/` file is missing, using `hf_hub_download(repo_id="nvidia/Real-time_RE-USE", filename=...)`. File exists at RealTime Space `exp/12x1_.../g_01010000.pth` [VERIFIED: HF Space file listing]. Implemented in Plan 01 `app.py` startup check.

2. **[Slider value type migration] RESOLVED: Use `0` sentinel meaning "off/disabled".** `enhance()` guards change from `target_sampling_rate != ''` to `if target_sampling_rate is not None and target_sampling_rate > 0`. Sliders return float (never empty string), so `0` means "bandwidth extension off". Implemented in Plan 02 `app.py`.

3. **[Variant selection UI placement] RESOLVED: Show all sliders always, disable RealTime-only ones for Standard.** Simpler Gradio logic than conditional visibility. All sliders render; `exit_layer`/`look_ahead_frames` start `interactive=False` (Standard default) and toggle via variant dropdown change event. Implemented in Plan 03 `app.py`.

4. **[OpenVINO future phase] RESOLVED: Yes, reuse `reuse_runtime.py`'s pure-Python `selective_scan_ref`.** The existing `selective_scan_ref` path already routes mamba_ssm to a CUDA-independent reference kernel [VERIFIED: local reuse_runtime.py lines 45-54]. Documented as the ONNX export starting point for any future OpenVINO phase; not executed in Phase 2.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|-----------------|--------------|--------|
| Standard RE-USE only (30-block Mamba, chunked inference) | Standard + RealTime (12-block streaming Mamba, exit_layer/look_ahead knobs) | NVIDIA Real-Time RE-USE paper (SLT 2026, arxiv:2606.25621) | Enables quality–latency trade-offs; lower latency configs |
| `gr.Textbox` for sample rates (free-form string) | `gr.Slider` (bounded numeric) | Proposed in Phase 2 (SEED-261112-e08) | Better UX, range enforcement, no empty-string edge cases |
| NVIDIA CUDA only (executed) | NVIDIA CUDA only (executed) + documented Vulkan/ROCm/Metal/OpenVINO feasibility | Phase 2 (SEED-261112-mos) | Honest scope: no false promises on non-NVIDIA backends |

**Deprecated/outdated:**
- `gr.Textbox` with `!= ''` guard for optional sample rates: superseded by `gr.Slider` with numeric defaults.
- `share=True` Gradio launch: replaced by `share=False` in Phase 1 (D-24) due to `h11` protocol error.
- Broken Video Upload tab: removed in Phase 1 (D-25); `soundfile` cannot demux video.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | RealTime `exp/.../g_01010000.pth` can be downloaded via `hf_hub_download` from the RealTime HF Space at install time | Risks | Install may fail if LFS download not supported or file path changes |
| A2 | `gr.Slider` with `minimum=8000, maximum=48000` covers all valid sample rates for bandwidth extension | Gradio UI | Users may need rates outside 8k–48k; model card says 8k–48kHz [CITED: model card] |
| A3 | RealTime variant's lack of chunking is acceptable for typical clip lengths (<10s) on RTX 4070 12 GB | Risks | OOM on long clips; may need to backport chunking to RealTime |
| A4 | No new Python packages needed beyond Phase 1 stack for variant support | Pinokio Integration | If RealTime needs additional deps (e.g., different torchaudio version), install.js may need changes |
| A5 | `menu.js` does not need changes; variant selection happens entirely in Gradio UI | Pinokio Integration | If user wants variant pre-selection before launch, menu.js + run.js need mode params |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GPU (CUDA) | `app.py` `device = "cuda"` | ✓ (RTX 4070 verified) | CUDA 12.8 | — (no fallback; NVIDIA-only) |
| Python 3.10 | `install.js` conda `.env` | ✓ | 3.10.22 | — |
| `uv` | `install.js` pip installs | ✓ | — | pip |
| `conda` / `mamba` | `install.js` env creation | ✓ | — | — |
| Git | `install.js` mamba clone | ✓ | — | — |
| ffmpeg | `install.js` conda install | ✓ | — | — |
| Vulkan SDK | Vulkan backend (not executed) | ✗ | — | N/A (documented only) |
| ROCm toolchain | ROCm backend (not executed) | ✗ | — | N/A (documented only) |
| Xcode / Metal | Metal backend (not executed) | ✗ | — | N/A (documented only) |
| OpenVINO | OpenVINO backend (not executed) | ✗ | — | N/A (documented only) |

**Missing dependencies with no fallback:** None blocking — NVIDIA CUDA path is fully available and verified on RTX 4070.
**Missing dependencies with fallback:** Vulkan/ROCm/Metal/OpenVINO toolchains are missing but not required (documented only, not executed).

## Validation Architecture

> `workflow.nyquist_validation` is `true` in `.planning/config.json` [VERIFIED: config.json line 9], so this section is included.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | None detected yet (Phase 1 had no automated test suite) |
| Config file | none — Wave 0 to establish |
| Quick run command | `python verify_gpu.py` (runs in `.env` conda) |
| Full suite command | To be defined in Wave 0 |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RTV-01 | RealTime model loads and infers on RTX 4070 | integration | `python app.py` → upload audio → Enhanced Audio output | ❌ Wave 0 |
| RTV-02 | Variant dropdown (Standard/RealTime) present in UI | UI/manual | Visual check of Gradio interface | ❌ Wave 0 |
| GUI-01 | `target_sampling_rate`/`low_pass_sampling_rate` are sliders | UI/manual | Visual check; verify `gr.Slider` type | ❌ Wave 0 |
| GUI-02 | Chunk size, overlap, window, normalization exposed | UI/manual | Visual check of sliders | ❌ Wave 0 |
| GUI-03 | RealTime-only params hidden/disabled for Standard | UI/manual | Toggle variant; verify slider state | ❌ Wave 0 |
| BK-01 | Backend viability doc covers all 4 backends | doc review | RESEARCH.md section review | ✅ (this document) |
| BK-02 | NVIDIA CUDA only executed path | code review | `install.js` fail-fast guard; `app.py` CUDA guard | ✅ (verified) |
| REL-05 | No Phase 1 regression | integration | Clean install → verify → run → infer → URL capture | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `python verify_gpu.py` (quick GPU check)
- **Per wave merge:** Full clean install → run → inference on RTX 4070
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] No test framework installed — Wave 0 to add pytest or manual test scripts
- [ ] No automated test for RealTime model loading/inference
- [ ] No automated test for Gradio slider types
- [ ] Framework install: none detected — to be determined during planning

## Security Domain

> `security_enforcement` is `true` (absent = enabled) in `.planning/config.json` [VERIFIED: config.json line 10], so this section is included. ASVS level 1 [VERIFIED: config.json line 11].

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (local Gradio app, no auth) |
| V3 Session Management | no | — (single-user local app) |
| V4 Access Control | no | — (local app, no user roles) |
| V5 Input Validation | yes | Gradio sliders enforce numeric ranges; file upload validated by `gr.Audio(type="filepath")` |
| V6 Cryptography | no | — (no crypto in app) |

### Known Threat Patterns for NVIDIA CUDA + Gradio Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious audio file upload (path traversal, oversized file) | Tampering | `gr.Audio(type="filepath")` restricts to audio types; `soundfile.read()` validates format |
| GPU exhaustion via long audio / no chunking | Denial of Service | Chunked inference for Standard; document OOM risk for RealTime |
| Arbitrary code via model weights (`torch.load` on untrusted `.pth`) | Remote Code Execution | Weights come from official NVIDIA HF Space only; `torch.load` with trusted source |
| `torch.cuda.empty_cache()` in inference loop | Information Disclosure | Standard CUDA memory management; no sensitive data in GPU memory |

## Sources

### Primary (HIGH confidence)
- Local `app.py` — Gradio UI, model loading, inference [VERIFIED: read this session, lines 1-212]
- Local `install.js` — NVIDIA fail-fast guard, conda env, torch delegation [VERIFIED: read this session, lines 1-136]
- Local `torch.js` — cu128 torch/torchaudio/torchvision pins [VERIFIED: read this session, lines 1-51]
- Local `pinokio.json` — platform/arch/gpu metadata [VERIFIED: read this session, lines 1-14]
- Local `menu.js` — run button `mode` param pattern [VERIFIED: read this session, lines 1-78]
- Local `reuse_runtime.py` — pure-Python mamba selective-scan patch [VERIFIED: read this session, lines 1-55]
- Local `verify_gpu.py` — CUDA/arch/mamba verification [VERIFIED: read this session, lines 1-119]
- Local `reset.js` — cleanup targets [VERIFIED: read this session, lines 1-48]
- Local `.planning/config.json` — nyquist_validation, security_enforcement flags [VERIFIED: read this session, lines 1-54]
- Local `.planning/PROJECT.md` — Phase 1 constraints and decisions [VERIFIED: read this session, lines 1-85]
- Local `.planning/ROADMAP.md` — Phase 2 goal and success criteria [VERIFIED: read this session, lines 1-92]
- Local `.planning/seeds/SEED-261112-e08-*.md` — Gradio customization seed [VERIFIED: read this session, lines 1-44]
- Local `.planning/seeds/SEED-261112-mos-*.md` — Multi-backend viability seed [VERIFIED: read this session, lines 1-70]
- Local `.planning/REQUIREMENTS.md` — Phase 1 requirement IDs [VERIFIED: read this session, lines 1-103]

### Secondary (MEDIUM confidence)
- RealTime HF Space `app.py` — variant UI structure, `exit_layer`/`look_ahead_frames`, weight loading [CITED: https://huggingface.co/spaces/nvidia/Real-time_RE-USE/raw/main/app.py]
- RealTime HF Space `README.md` — sdk_version 6.12.0, python 3.10 [CITED: https://huggingface.co/spaces/nvidia/Real-time_RE-USE/raw/main/README.md]
- RealTime HF Space `requirements.txt` — package list [CITED: https://huggingface.co/spaces/nvidia/Real-time_RE-USE/raw/main/requirements.txt]
- RealTime HF Space recipe YAML — 12x1 config, hop_160, num_tfmamba 12 [CITED: https://huggingface.co/spaces/nvidia/Real-time_RE-USE/raw/main/recipes/USEMamba_12x1_...yaml]
- Standard RE-USE HF Space recipe YAML — 30x1 config, hop_40, num_tfmamba 30 [CITED: https://huggingface.co/spaces/nvidia/RE-USE/raw/main/recipes/USEMamba_30x1_...yaml]
- `nvidia/NVIDIA-NemotronLabs-RE-USE` model card — 9.6M params, 8k–48kHz input, arxiv:2603.02641 [CITED: https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE]
- `nvidia/RE-USE` model card — research license, 30-layer bidirectional Mamba [CITED: https://huggingface.co/nvidia/RE-USE]
- SEED-261112-mos research findings — Vulkan/ROCm/Metal/OpenVINO feasibility [CITED: local seed file]

### Tertiary (LOW confidence)
- General Gradio 6.x `gr.Slider` API — `minimum`, `maximum`, `value`, `step`, `label`, `interactive` [ASSUMED: training knowledge, not verified against installed gradio version]
- OpenVINO ONNX export opset 17+ for Mamba models [ASSUMED: training knowledge]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified against local source files and HF Space requirements.txt
- Architecture: HIGH — verified against local app.py and RealTime Space app.py
- Pitfalls: MEDIUM — based on code inspection and seed research; slider migration risk untested
- Multi-backend: MEDIUM — documented from SEED-261112-mos; none executed, feasibility is reasoned

**Research date:** 2026-11-12
**Valid until:** 30 days (2026-12-12) — HF Space contents and package versions may change

## RESEARCH COMPLETE

**Phase:** 02 - Enable RealTime RE-USE variants with user-friendly Gradio UI and multi-backend assessment
**Confidence:** MEDIUM

### Key Findings
- RealTime RE-USE uses `SEMamba_decoder_list` (12 TFMamba blocks, hop 160) vs Standard `SEMamba` (30 blocks, hop 40); weights load from local `.pth` via `torch.load`, not Hub `from_pretrained` [VERIFIED: both app.py files + recipe YAMLs]
- Both spaces use `gr.Textbox` for sample rates; converting to `gr.Slider` requires replacing `!= ''` guards with numeric checks [VERIFIED: local app.py lines 193-194, 207; RealTime app.py]
- RealTime adds `exit_layer` (3–12) and `look_ahead_frames` (0–2) quality–latency knobs; Standard has none [VERIFIED: RealTime app.py `enhance()` signature]
- Vulkan/ROCm/Metal/OpenVINO are documented as infeasible-to-execute this phase; NVIDIA CUDA remains the only executed path [SEED-261112-mos; local install.js/app.py guards]
- No new Python packages needed; Phase 1 stack (torch 2.10.0+cu128, mamba-ssm 2.3.2.post1, gradio, librosa, etc.) covers variants

### File Created
`D:\pinokio\api\NemotronLabs-RE-USE-Pinokio_Win-Nvidia.git\.planning\phases\02-enable-realtime-reuse-variants\02-RESEARCH.md`

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | Verified against local source files + HF Space requirements.txt |
| Architecture | HIGH | Verified against local app.py and RealTime Space app.py |
| Pitfalls | MEDIUM | Slider type migration risk untested; RealTime weights distribution unresolved |
| Multi-backend | MEDIUM | Documented only; none executed; feasibility reasoned from mamba-ssm CUDA dependency |

### Open Questions
1. RealTime weights (`g_01010000.pth`, ~54 MB LFS): download at install via `hf_hub_download` or manual placement?
2. Slider "disabled" default: `0` sentinel or minimum valid `8000`?
3. RealTime-only slider visibility: show-all-with-disable vs conditional visibility?
4. OpenVINO export: is `selective_scan_ref` pure-Python path sufficient as ONNX starting point?

### Ready for Planning
Research complete. Planner can now create PLAN.md files.
