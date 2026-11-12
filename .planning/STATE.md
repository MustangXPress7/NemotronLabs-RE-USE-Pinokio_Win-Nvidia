---
gsd_state_version: "1.0"
current_phase: 2
current_phase_name: Enable RealTime RE-USE variants
status: complete
stopped_at: Completed 02-enable-realtime-reuse-variants-01-PLAN.md
last_updated: "2026-11-12T14:40:37.885Z"
last_activity: 2026-11-12
last_activity_desc: Phase 2 execution started
state_head: 103a1cbc365c0c9944d1291fd341d58858008552
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 10
  completed_plans: 6
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-11-09)

**Core value:** A clean install on a supported NVIDIA machine produces a working GPU inference environment and a running Gradio app, without hidden manual steps and without documented claims that were never verified.
**Current focus:** Phase 2 — Enable RealTime RE-USE variants with user-friendly Gradio UI and multi-backend assessment (planned, ready to execute)

## Current Position

Phase: 2 of 2 (Enable RealTime RE-USE variants)
Plan: 1 of 5 in current phase
Status: In Progress - executing Wave 1/5 (02-01: variant registry + RealTime model load)
Last activity: 2026-11-12 - Phase 2 execution started

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: single session
- Total execution time: one session

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. NVIDIA install and GPU truth | 5 | 5 | one session |
| 2. Enable RealTime RE-USE variants... | 0 | 5 | planned |

**Recent Trend:**
- Last 5 plans: 01-01 … 01-05
- Trend: Complete → Phase 2 planned


*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 02-enable-realtime-reuse-variants P01 | 25min | 1 tasks | 1 files |

## Accumulated Context

### Decisions

- Phase 1: NVIDIA-only GPU scope; no AMD/Apple/CPU promises
- Phase 1: Validate on RTX 4070 only; Quadro RTX 6000 compatibility is reasoned from `sm_75`
- Phase 1: One phase, five plans, ending with a verification report
- Phase 1: Keep the conda `.env` with Python 3.10 and relative launcher paths
- Phase 1: Pin `torch 2.10.0` / `torchvision 0.25.0` / `torchaudio 2.10.0` cu128
- Phase 1: Build mamba `v2.3.2.post1` with `MAMBA_SKIP_CUDA_BUILD=TRUE` only
- Phase 1: Do not install `causal-conv1d`; it is optional and broken on this machine
- Phase 1: `app.py` guards CUDA, launches with `share=False`, drops the video tab
- Phase 1: `run.js` stays unchanged; `input.event[1]` still captures the URL
- Phase 2: Model variant registry pattern (Standard/RealTime) in app.py, no scattered if/else
- Phase 2: NVIDIA CUDA remains the only executed path; Vulkan/ROCm/Metal/OpenVINO documented only
- Phase 2: Gradio sliders replace Textboxes; `0` sentinel means "off/disabled"
- Phase 2: RealTime-only sliders (exit_layer, look_ahead_frames) disabled for Standard variant
- Phase 2: No new Python packages beyond Phase 1 stack
- [Phase 02-enable-realtime-reuse-variants]: Variant registry pattern instead of scattered if/else chains — single lookup point, easy to add a third variant later
- [Phase 02-enable-realtime-reuse-variants]: Keep device='cuda' and share=False untouched (Phase 1 guards D-01/D-24 preserved)
- [Phase 02-enable-realtime-reuse-variants]: RealTime weights load via torch.load from local exp/.../g_01010000.pth (not from_pretrained Hub)
- [Phase 02-enable-realtime-reuse-variants]: Standard path unchanged: SEMamba + hop_40 recipe + from_pretrained nvidia/NVIDIA-NemotronLabs-RE-USE

### Pending Todos

- Run `Install` from a clean state (Reset first) to close INST-01, VER-03, and ROADMAP Criterion 1 (Phase 1 open item).
- Execute Phase 2 plans 02-01 through 02-05 in wave order on RTX 4070.
- Verify RealTime variant inference on RTX 4070 (Plan 02-01 tracer slice).
- Verify Phase 1 regression after all app.py changes (Plan 02-04).

### Blockers/Concerns

- Phase 1: The clean install from empty `.env` still needs to be re-executed (INST-01, VER-03).
- Phase 2: RealTime weights distribution — download via `hf_hub_download` at startup (resolved in research, untested in execution).

## Deferred Items

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| GPU | AMD ROCm support | Deferred | 2026-11-09 | v1 |
| GPU | Apple Silicon / MPS | Deferred | 2026-11-09 | v1 |
| Inference | CPU fallback | Deferred | 2026-11-09 | v1 |
| Docs | Quadro RTX 6000 benchmark numbers | Out of scope | 2026-11-09 | v1 |

## Session Continuity

Last session: 2026-11-12T14:40:28.725Z
Stopped at: Completed 02-enable-realtime-reuse-variants-01-PLAN.md
Resume file: `.planning/phases/02-enable-realtime-reuse-variants/02-RESEARCH.md`
Next: `/gsd-execute-phase 02` to execute plans 02-01 through 02-05 in wave order
