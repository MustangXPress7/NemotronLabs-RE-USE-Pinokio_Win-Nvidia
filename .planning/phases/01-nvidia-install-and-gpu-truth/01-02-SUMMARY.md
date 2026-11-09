---
phase: 01-nvidia-install-and-gpu-truth
plan: 02
subsystem: install
tags: [pinokio, conda, mamba, torch, reset]
requires:
  - plan: 01-01
    provides: Verdicts and the pinned mamba/torch facts
provides:
  - Declarative .env with Python 3.10
  - NVIDIA fail-fast guard before any download
  - Single live mamba build flag
  - Artifact-only reset
affects: [torch, verification, docs]
key-files:
  created: []
  modified: [install.js, reset.js]
key-decisions:
  - "Use relative paths (.env, mamba, .) in launcher scripts"
  - "Set only MAMBA_SKIP_CUDA_BUILD for the pinned tag"
  - "Reset keeps output/ and logs/"
requirements-completed: [INST-01, INST-02, INST-03, INST-05, INST-06]
duration: single session
completed: 2026-11-09
status: complete
---

# Plan 02 Summary: Harden install and reset

`install.js` now has 11 steps: GPU/platform guard, declarative `.env` with Python 3.10,
FFmpeg, pip upgrade, `torch.js`, requirements, incomplete-clone cleanup, pinned mamba
clone, pure-Python mamba build with only `MAMBA_SKIP_CUDA_BUILD`, `python verify_gpu.py`,
and a success input step. `requires.bundle: "ai"` is preserved.

`reset.js` now has 5 steps and removes only `.env`, `cache`, `mamba`, and the legacy
`causal-conv1d` folder. It no longer deletes `logs/` or `output/`.

Both files pass `node -e require(...)`. The new install flow itself was not run from a
clean state; that remains the single open verification item.
