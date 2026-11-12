---
phase: 02-enable-realtime-reuse-variants
plan: 02
subsystem: ui
tags: [gradio, mamba, speech-enhancement, nvidia, cuda, model-variants, realtime, sliders]

# Dependency graph
requires:
  - phase: 02-enable-realtime-reuse-variants
    plan: 01
    provides: VARIANT_REGISTRY dict, Model Variant dropdown, RealTime model load + variant-aware inference
provides:
  - gr.Slider for target_sampling_rate (8000-48000 Hz, default 0 off)
  - gr.Slider for low_pass_sampling_rate (8000-48000 Hz, default 0 off)
  - gr.Slider for chunk_size_s (1-10 seconds, default 3)
  - gr.Slider for overlap_portion (0.0-0.9, default 0.5)
  - gr.Dropdown for window_type (Hann/Hamming, default Hann)
  - gr.Slider for normalization (0.0-1.0, default 1.0)
  - Numeric guard logic (`is not None and > 0`) replacing `!= ''` empty-string checks
  - enhance() signature: (filepath, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization)
affects: [02-03, 02-04, 02-05]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
# Same estimateTokens scale (chars/4 over the realized diff), never a harness token count.
actuals:
  tokens: 400
  tasks: 1
  commits: 1

# Tech tracking
tech-stack:
  added: []
  patterns: [gr.Slider bounded numeric inputs, slider-driven inference parameters, numeric guard logic]

key-files:
  created: []
  modified:
    - app.py - gr.Slider conversions for sample rates + chunk/overlap/window/normalization sliders wired into enhance()

key-decisions:
  - "Slider value 0 used as sentinel meaning 'off/disabled' for optional sample rates (sliders never return empty string)"
  - "Window type exposed as gr.Dropdown (categorical Hann/Hamming) rather than gr.Slider"
  - "Normalization scales the window-sum divisor: audio_enhanced / torch.clamp(window_sum * normalization, min=1e-7)"
  - "enhance_btn.click inputs list expanded to 8 entries: [input_audio, lp_sr, target_sr, variant, chunk, overlap, window, norm]"

requirements-completed: [GUI-01, GUI-02]

# Metrics
duration: ~20min
completed: 2026-11-12
status: complete
plan_head_before: 1f1f7d8
plan_head_after: 1f4a4f3fa4ad3309dbc4fb72014255dfa0ee53c8
commits: 1

# Phase 02 Plan 02: Gradio Slider Conversion Summary

**Gradio sliders replace Textboxes for sample rates; chunk/overlap/window/normalization exposed and wired into enhance().**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-11-12
- **Tasks:** 1
- **Files modified:** 1 (app.py)
- **Commits:** 1 (measured from plan head)

## Accomplishments

- Converted `target_sampling_rate` and `low_pass_sampling_rate` from `gr.Textbox` to `gr.Slider(minimum=8000, maximum=48000, value=0, step=1000, interactive=True)`
- Added `gr.Slider` for chunk size (1-10s, default 3), overlap portion (0.0-0.9, default 0.5), normalization (0.0-1.0, default 1.0)
- Added `gr.Dropdown` for window type (Hann/Hamming, default Hann)
- Updated `enhance()` signature to accept `chunk_size_s`, `overlap_portion`, `window_type`, `normalization`
- Replaced hardcoded `chunk_size = int(3.0 * noisy_sr)` with `chunk_size = int(chunk_size_s * noisy_sr)`
- Replaced hardcoded `hop_length = int(0.5 * chunk_size)` with `hop_length = int(overlap_portion * chunk_size)`
- Replaced `torch.hann_window(chunk_size)` with window-type selection (Hann->torch.hann_window, Hamming->torch.hamming_window, clamped min=1e-7)
- Replaced final normalization with `audio_enhanced / torch.clamp(window_sum * normalization, min=1e-7)`
- Replaced `target_sampling_rate != ''` / `low_pass_sampling_rate != ''` with numeric `is not None and > 0` guards
- Updated `unified_enhance()` and `enhance_btn.click` inputs list to 8 entries

## Task Commits

Each task was committed atomically:

1. **Task 1: Convert sample-rate Textboxes to Sliders and add chunk/overlap/window/normalization sliders** - `1f4a4f3` (feat)

**Plan metadata:** `1f4a4f3fa4ad3309dbc4fb72014255dfa0ee53c8` (feat: complete plan)

## Files Created/Modified

- `app.py` - gr.Slider conversions for sample rates + chunk/overlap/window/normalization sliders wired into enhance()

## Decisions Made

- Slider value 0 used as sentinel meaning 'off/disabled' for optional sample rates (sliders never return empty string)
- Window type exposed as gr.Dropdown (categorical Hann/Hamming) rather than gr.Slider
- Normalization scales the window-sum divisor
- enhance_btn.click inputs list expanded to 8 entries preserving variant dropdown position

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all slider values flow into enhance() and affect inference.

## Threat Flags

None - no new security-relevant surface introduced (sliders enforce numeric ranges; no new endpoints or file access patterns).

## User Setup Required

None - no external service configuration required.

## Self-Check: PASSED

- `app.py` exists: FOUND
- Task commit `1f4a4f3fa4ad3309dbc4fb72014255dfa0ee53c8` is ancestor of HEAD: YES
- Python syntax check (`ast.parse`): exits 0
- `gr.Slider` count: 5 (multiple occurrences)
- `gr.Textbox` count: 0 (no Textboxes remain)
- `!= ''` count: 0 (no empty-string guards remain)
- `chunk_size_s` parameter in enhance() signature: YES
- `overlap_portion` parameter in enhance() signature: YES
- `normalization` parameter in enhance() signature: YES
- `window_type` parameter in enhance() signature: YES
- `enhance_btn.click` inputs list contains 8 entries: YES

---
*Phase: 02-enable-realtime-reuse-variants*
*Completed: 2026-11-12*
