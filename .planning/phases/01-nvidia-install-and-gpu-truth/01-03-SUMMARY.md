---
phase: 01-nvidia-install-and-gpu-truth
plan: 03
subsystem: gpu-runtime
tags: [nvidia, cuda, pytorch, mamba, gradio]
requires:
  - plan: 01-02
    provides: Install flow and pinned package versions
provides:
  - NVIDIA-only torch.js with fail-fast guard
  - Shared mamba selective-scan patch
  - Repeatable verify_gpu.py and verify.js menu entry
  - CUDA-guarded, audio-only, local Gradio app
affects: [verification, docs, menu]
key-files:
  created: [reuse_runtime.py, verify_gpu.py, verify.js]
  modified: [torch.js, app.py, menu.js]
key-decisions:
  - "torch.js keeps only Windows/Linux NVIDIA cu128 branches"
  - "One shared patch in reuse_runtime.py is applied by app and verifier"
  - "app.py raises SystemExit without CUDA and launches with share=False"
  - "Video tab removed; input is audio-only"
requirements-completed: [GPU-01, GPU-02, GPU-03, GPU-04, REL-02, REL-04]
duration: single session
completed: 2026-11-09
status: complete
---

# Plan 03 Summary: Explicit NVIDIA GPU support

`torch.js` was reduced to two cu128 branches (Windows and Linux) plus a final
`break: true` guard for anything else. AMD, Apple, Intel Mac, and CPU branches were
removed.

`reuse_runtime.py` centralizes the mamba patch and the reference-kernel binding.
`verify_gpu.py` reports JSON and exits non-zero on failure; `verify.js` runs it inside
`.env`; `menu.js` exposes **Verify GPU**.

`app.py` now imports the helper, raises `SystemExit` when CUDA is unavailable, drops the
broken video tab, uses `share=False`, and launches Gradio only under
`if __name__ == "__main__":`. `run.js` is unchanged.
