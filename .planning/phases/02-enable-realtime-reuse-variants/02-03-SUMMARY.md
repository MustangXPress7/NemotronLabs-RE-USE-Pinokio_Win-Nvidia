---
phase: 02-enable-realtime-reuse-variants
plan: 03
subsystem: ui
tags: [gradio, mamba, speech-enhancement, nvidia, cuda, model-variants, realtime, sliders]

# Dependency graph
requires:
  - phase: 02-enable-realtime-reuse-variants
    plan: 02
    provides: gr.Slider for sample rates/chunk/overlap/window/normalization, variant-aware enhance() with 8 inputs
provides:
  - gr.Slider exit_layer (3-12, default 8, RealTime only)
  - gr.Slider look_ahead_frames (0-2, default 0, RealTime only)
  - Variant dropdown change handler toggling slider interactivity
  - enhance() signature: (filepath, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer=8, look_ahead_frames=0)
  - enhance_btn.click inputs expanded to 10 entries
affects: [02-04, 02-05]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
# Same estimateTokens scale (chars/4 over the realized diff), never a harness token count.
actuals:
  tokens: 1500
  tasks: 1
  commits: 1

# Tech tracking
tech-stack:
  added: []
  patterns: [gr.Slider bounded numeric inputs, variant-dependent UI enable/disable, variant-aware inference dispatch]

key-files:
  created: []
  modified:
    - app.py - RealTime-only sliders with variant-dependent enable/disable, variant-aware enhance() dispatch

key-decisions:
  - "Both RealTime-only sliders start interactive=False since Standard is the default variant"
  - "Variant dropdown change event uses gr.update(interactive=(variant == 'RealTime')) to toggle both sliders"
  - "enhance() reads exit_layer/look_ahead_frames from function parameters (not registry defaults)"
  - "Standard variant calls model(mag, pha) without layer_use/number_ahead; unchanged Phase 1 behavior"

metrics:
  duration: ~15min
  completed: 2026-11-12
  status: complete
  plan_head_before: 207ddf789abe41ac11c8944bf6cfe6a1a8427461
  plan_head_after: 2bf6a554af29088877dd4cd9d3b59529380bc777
  commits: 1

---

# Phase 02 Plan 03: RealTime-Only exit_layer/look_ahead_frames Sliders Summary

**RealTime-only sliders with variant-dependent enable/disable and variant-aware enhance() dispatch.**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-11-12
- **Tasks:** 1
- **Files modified:** 1 (app.py)
- **Commits:** 1 (measured from plan head)

## Accomplishments

- Added `gr.Slider` for exit_layer (minimum=3, maximum=12, value=8, step=1, label="Exit layer (RealTime only)", interactive=False)
- Added `gr.Slider` for look_ahead_frames (minimum=0, maximum=2, value=0, step=1, label="Look ahead frames (RealTime only)", interactive=False)
- Added `variant_dropdown.change` handler using `gr.update(interactive=(variant == "RealTime"))` to toggle both sliders
- Updated `enhance()` signature to `(filepath, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer=8, look_ahead_frames=0)`
- Removed `entry.get("exit_layer_default", 8)` / `entry.get("look_ahead_frames_default", 0)` — parameters now flow from UI sliders
- `enhance()` passes `layer_use=exit_layer, number_ahead=look_ahead_frames` to RealTime model; Standard calls `model(mag, pha)` unchanged
- Updated `unified_enhance()` signature and body to forward exit_layer/look_ahead_frames
- Updated `enhance_btn.click` inputs list to 10 entries

## Task Commits

Each task was committed atomically:

1. **Task 1: Add exit_layer/look_ahead_frames sliders with variant-dependent visibility + RealTime-only dispatch in enhance()** - `2bf6a55` (feat)

**Plan metadata:** `2bf6a554af29088877dd4cd9d3b59529380bc777` (feat: complete plan)

## Files Created/Modified

- `app.py` - RealTime-only sliders with variant-dependent enable/disable, variant-aware enhance() dispatch

## Decisions Made

- Both RealTime-only sliders start disabled (Standard is default variant)
- Variant dropdown change event toggles slider interactivity via `gr.update()`
- enhance() uses function parameters for exit_layer/look_ahead_frames instead of registry defaults

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all slider values flow into enhance() and affect inference per variant.

## Threat Flags

None - no new security-relevant surface introduced (sliders enforce numeric ranges; no new endpoints or file access patterns).

## User Setup Required

None - no external service configuration required.

## Self-Check: PASSED

- `app.py` exists: FOUND
- Task commit `2bf6a554af29088877dd4cd9d3b59529380bc777` is ancestor of HEAD: YES
- Python syntax check (`ast.parse`): exits 0
- `exit_layer` count: 10 (>= 1)
- `look_ahead_frames` count: 10 (>= 1)
- `layer_use` count: 2 (>= 1)
- `number_ahead` count: 2 (>= 1)
- `interactive=False` count: 2 (>= 1)
- `variant_id == "realtime"` count: 1 (>= 1)
- `enhance_btn.click` inputs list contains 10 entries: YES
- `enhance()` signature has `exit_layer=8`: YES
- `enhance()` signature has `look_ahead_frames=0`: YES

---
*Phase: 02-enable-realtime-reuse-variants*
*Completed: 2026-11-12*
