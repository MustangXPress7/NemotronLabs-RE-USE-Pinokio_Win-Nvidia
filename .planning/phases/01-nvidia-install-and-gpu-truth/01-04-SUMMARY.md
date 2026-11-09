---
phase: 01-nvidia-install-and-gpu-truth
plan: 04
subsystem: docs
tags: [readme, gpu-matrix, learnings]
requires:
  - plan: 01-02
    provides: Final install and reset behavior
  - plan: 01-03
    provides: Final GPU and app behavior
provides:
  - Honest compatibility matrix (Tested / Reasoned / Unsupported / Unverified)
  - Learned and unlearned record
  - Script-accurate install and run instructions
affects: [distribution, support]
key-files:
  created: []
  modified: [README.md, README_PINOKIO.md]
key-decisions:
  - "Delete the fabricated Quadro RTX 6000 (22.5 GB) table"
  - "State plainly that no VRAM benchmark was run"
  - "Document MAMBA_SKIP_CUDA_BUILD and the correct torchvision pin"
requirements-completed: [DOC-01, DOC-02, DOC-03, DOC-04, GPU-05]
duration: single session
completed: 2026-11-09
status: complete
---

# Plan 04 Summary: Correct the documentation

`README.md` was rewritten around verified facts. It now has a Tested / Reasoned /
Unsupported / Unverified matrix, script-accurate install steps, a verification section
with the exact `verify_gpu.py` JSON keys, and a "What we learned / What we unlearned"
section covering arch coverage, the torchvision pin, the mamba flag mismatch,
`causal-conv1d`, and the `.m4a` limitation.

The fabricated "Measured on a Quadro RTX 6000 (22.5 GB)" table and the dead
`MAMBA_KEEP_CUDA_BUILD` references were removed. The memory section states that no VRAM
benchmark exists.

`README_PINOKIO.md` is now a short launcher note that points to `README.md` as the
single source of truth and lists the files that actually exist.
