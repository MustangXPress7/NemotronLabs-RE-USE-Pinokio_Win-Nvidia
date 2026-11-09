---
phase: 01-nvidia-install-and-gpu-truth
plan: 05
subsystem: verification
tags: [cuda, inference, gradio, evidence]
requires:
  - plan: 01-02
    provides: Installed environment and scripts
  - plan: 01-03
    provides: Verifier and guarded app
  - plan: 01-04
    provides: Documented claims to check
provides:
  - Executed GPU verification on the RTX 4070
  - End-to-end inference smoke test
  - Captured local Gradio URL
  - Verification report with reasoned/unverified list
affects: [release, support]
key-files:
  created:
    - verify_inference.py
    - .planning/phases/01-nvidia-install-and-gpu-truth/01-VERIFICATION.md
  modified: [.planning/REQUIREMENTS.md, .planning/ROADMAP.md, .planning/STATE.md]
key-decisions:
  - "Report status is human_needed because install.js was not re-run from a clean state"
  - "Quadro RTX 6000 stays reasoned, never measured"
  - "No VRAM numbers are published"
requirements-completed: [REL-01, VER-01, VER-02]
duration: single session
completed: 2026-11-09
status: complete
---

# Plan 05 Summary: Prove the result

Added `verify_inference.py`, which generates a deterministic 3-second 16 kHz WAV and runs
the real model through `app.enhance`.

Executed evidence on the RTX 4070:

- `python verify_gpu.py` → `ok: true`, CUDA available, arch list contains `sm_75` and
  `sm_86`, mamba 2.3.2.post1 bound to `selective_scan_ref`.
- `python verify_inference.py` → `ok: true`, 48,000 output frames, 4.28 s chunk timer,
  20.34 s wall time including first model load.
- `python app.py` → `Running on local URL: http://127.0.0.1:7860` in 8.1 s with no share
  attempt.

`01-VERIFICATION.md` records all truths, artifacts, key links, the 22 requirement
statuses, and the reasoned/unverified list. The one open behavior item is the clean
install: `install.js` has not been run from an empty `.env`, so INST-01, VER-03, and
ROADMAP Criterion 1 remain open.
