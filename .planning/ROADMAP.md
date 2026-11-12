# Roadmap: RE-USE Pinokio Launcher

## Overview

One phase to turn a set of unverified prior-AI edits into a reproducible, honest NVIDIA-only
launcher. The phase audits what the edits actually changed, hardens the install flow, makes GPU
support explicit and verifiable, corrects documentation, and proves the result on the available
RTX 4070 while marking Quadro RTX 6000 as reasoned-only.

## Phases

- [x] **Phase 1: NVIDIA install and GPU truth** - Audit prior edits, harden install, make NVIDIA CUDA support explicit and verified, fix docs, record learnings
- [ ] **Phase 2: Enable RealTime RE-USE variants with user-friendly Gradio UI and multi-backend assessment** - Incorporate SEED-261112-e08 (Gradio sliders for sample rate/chunk/overlap) and SEED-261112-mos (Vulkan/ROCm/Metal/OpenVINO viability), add the RealTime model variant from https://huggingface.co/spaces/nvidia/Real-time_RE-USE while keeping NVIDIA CUDA as the verified production path

## Phase Details

### Phase 1: NVIDIA install and GPU truth

**Goal**: Deliver a reproducible, NVIDIA-only launcher whose GPU support is explicit, verified on
an RTX 4070, and documented without unverified claims.

**Depends on**: Nothing (first phase)

**Requirements**: INST-01, INST-02, INST-03, INST-04, INST-05, INST-06, GPU-01, GPU-02, GPU-03,
GPU-04, GPU-05, REL-01, REL-02, REL-03, REL-04, DOC-01, DOC-02, DOC-03, DOC-04, VER-01, VER-02,
VER-03

**Success Criteria** (what must be TRUE):
  1. A clean install on Windows + RTX 4070 completes from scratch and produces `torch.cuda.is_available() == True` with an arch list containing `sm_75` and `sm_86`
  2. The app loads the model and enhances an audio file on the RTX 4070 after the changes, with `start.js` still capturing the Gradio URL
  3. Build flags and pinned versions in `install.js`/`torch.js` demonstrably affect the installed build — no ignored environment variables and no mismatched torch family versions
  4. README/docs contain no fabricated Quadro measurements, state Quadro RTX 6000 support as reasoned from `sm_75`, and record the learned/unlearned list
  5. A verification report records the executed checks, their evidence, and any remaining unverified claim

**Plans**: 5 plans

Plans:
- [x] 01-01: Audit prior edits and lock decisions (keep/revert/fix) with evidence
- [x] 01-02: Harden install/reset flow (venv, torch family pins, mamba flag, causal-conv1d, Python version)
- [x] 01-03: Make NVIDIA GPU support explicit (fail fast without GPU, arch check, metadata)
- [x] 01-04: Correct documentation and record learned/unlearned items
- [x] 01-05: Add verification checks, run validation on RTX 4070, write verification report

## Progress

**Execution Order:**
Phase 1 only.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. NVIDIA install and GPU truth | 5/5 | Complete | 2026-11-09 |

**Open item:** Success Criterion 1 is not closed. The new `install.js` has not been run from a
clean state; the existing `.env` was reused. See
`.planning/phases/01-nvidia-install-and-gpu-truth/01-VERIFICATION.md` ("Next Step to Close the Phase").

### Phase 2: Enable RealTime RE-USE variants with user-friendly Gradio UI and multi-backend assessment

**Goal**: Enable additional RE-USE model variants (starting with RealTime) in the Pinokio
launcher, incorporating SEED-261112-e08 (Gradio UI customization with sliders for sample rate,
chunk size, overlap) and SEED-261112-mos (Vulkan/ROCm/Metal/OpenVINO viability assessment), while
keeping NVIDIA CUDA as the verified production path.

**Depends on**: Phase 1 (NVIDIA install and GPU truth — must remain working)

**Requirements**: RTV-01, RTV-02, RTV-03, GUI-01, GUI-02, GUI-03, BK-01, BK-02, PIN-01, PIN-02, PIN-03,
REL-05 — model variant selection, Gradio UI sliders, backend viability documentation, RealTime model
loading and inference, NVIDIA CUDA-only execution, no Phase 1 regression

**Success Criteria** (what must be TRUE):
  1. The RealTime RE-USE model variant loads and performs inference on the RTX 4070 via the
     launcher's Gradio interface
  2. `app.py` exposes at least `target_sampling_rate` and `low_pass_sampling_rate` as Gradio
     sliders (not Textboxes) per SEED-261112-e08
  3. Additional model parameters (chunk size, overlap, window, normalization) are exposed from
     the Gradio UI per SEED-261112-e08
  4. A documented viability assessment covers Vulkan, ROCm, Metal, and OpenVINO backends per
     SEED-261112-mos, with NVIDIA CUDA remaining the only executed path
  5. `pinokio.json` metadata reflects the new model variant(s) and any newly supported platforms
  6. No regression on Phase 1: clean install, GPU verify, inference, and Gradio URL capture all
     still pass on RTX 4070

**Plans**: 5 plans
- [x] 02-01-PLAN.md
- [x] 02-02-PLAN.md
- [x] 02-03-PLAN.md
- [x] 02-04-PLAN.md
- [x] 02-05-PLAN.md

## Progress

**Execution Order:**
Phase 1 → Phase 2.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. NVIDIA install and GPU truth | 5/5 | Complete | 2026-11-09 |
| 2. Enable RealTime RE-USE variants... | 5/5 | Complete | 2026-11-12 |
