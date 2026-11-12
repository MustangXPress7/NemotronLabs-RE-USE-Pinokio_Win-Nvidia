---
phase: 02-enable-realtime-reuse-variants
plan: 04
subsystem: launcher
tags: [pinokio, metadata, regression, nvidia, cuda, model-variants, realtime]
requires:
  - phase: 01-nvidia-install-and-gpu-truth
    provides: NVIDIA-only launcher with working GPU verify, inference, and Gradio URL capture on RTX 4070 (Phase 1 path preserved by this plan)
  - phase: 02-enable-realtime-reuse-variants
    plan: 03
    provides: Variant registry + Gradio Model Variant dropdown in app.py (Wave 1-3 changes this plan's metadata documents and whose regression this plan re-verifies)
provides:
  - pinokio.json metadata declaring model-variants + realtime tags and Real-Time RE-USE Space homepage; NVIDIA scope unchanged (gpu [nvidia], platform [win32/linux], arch [x64])
  - Phase 1 regression evidence on RTX 4070: verify_gpu.py (CUDA + sm_75/sm_86), Gradio local URL capture (share=False), Standard variant enhance() -> 48,000-frame WAV
affects:
  - 02-enable-realtime-reuse-variants plan 05 (final plan in phase)
actuals:
  tokens: 4444
  tasks: 1
  commits: 1
tech-stack:
  added: []
  patterns:
    - Variant selection stays in the Gradio UI (not Pinokio menu level) per research A5
    - NVIDIA-only launcher metadata: pinokio.json gpu/platform/arch unchanged
key-files:
  created: []
  modified:
    - pinokio.json - launcher metadata: tags array + homepage
    - app.py - deviation fix: graceful variant recipe loading so the Standard Phase 1 path stays runnable
key-decisions:
  - "Homepage points to the Real-Time RE-USE Space as the variant reference; repository link stays https://github.com/NVIDIA/RE-USE"
  - "No menu.js or install.js changes: variant selection lives in app.py's Gradio dropdown; install stays NVIDIA-only"
requirements-completed: [PIN-01, PIN-02, PIN-03, REL-05]
coverage:
  - id: D1
    description: pinokio.json declares model-variants and realtime tags, homepage references Real-Time RE-USE Space, NVIDIA scope unchanged
    requirement: PIN-01
    verification:
      - kind: automated
        ref: 'python -c "import json; d=json.load(open(pinokio.json)); assert model-variants in d[tags]; assert realtime in d[tags]; assert d[gpu]==[nvidia]"'
        status: pass
    human_judgment: false
  - id: D2
    description: menu.js run button unchanged (passes mode Default to run.js, no variant pre-selection); install.js unchanged (NVIDIA fail-fast guard, torch.js delegation, no new deps)
    requirement: PIN-02 / PIN-03
    verification:
      - kind: automated
        ref: findstr checks on menu.js (mode, run.js), install.js (gpu !== nvidia), run.js (input.event[1])
        status: pass
    human_judgment: false
  - id: D3
    description: Phase 1 regression on RTX 4070: verify_gpu.py ok, app.py serves local Gradio URL with share=False, Standard enhance() yields WAV frames > 0
    requirement: REL-05
    verification:
      - kind: integration
        ref: .\.env\python.exe verify_gpu.py -> cuda_available true, arch contains sm_75 + sm_86
        status: pass
      - kind: integration
        ref: .\.env\python.exe app.py -> Running on local URL: http://127.0.0.1:7860, process terminated after URL capture
        status: pass
      - kind: integration
        ref: .\.env\python.exe verify_inference.py -> ok: true, output_frames: 48000
        status: pass
    human_judgment: false
---
# Phase 02 Plan 04: pinokio.json metadata + Phase 1 regression verify Summary

**pinokio.json tags add model-variants and realtime, homepage references the Real-Time RE-USE Space, and the Phase 1 regression (GPU verify, Gradio local URL, Standard enhance) re-verifies clean on RTX 4070.**

## Performance

- **Duration:** ~19 min
- **Started:** 2026-11-12T14:51:20Z
- **Completed:** 2026-11-12T15:10:00Z
- **Tasks:** 1 of 1 complete
- **Files modified:** 2 (pinokio.json, app.py)

## Accomplishments

- pinokio.json: tags extended with `model-variants` and `realtime`; homepage now `https://huggingface.co/spaces/nvidia/Real-time_RE-USE`; `gpu: ["nvidia"]`, `platform: ["win32","linux"]`, `arch: ["x64"]`, `version: "1.0"` all unchanged
- menu.js confirmed unchanged: run button passes `mode: 'Default'` to run.js; variant selection stays in the Gradio UI per research A5 (no menu-level variant pre-selection)
- install.js confirmed unchanged: NVIDIA fail-fast guard (`gpu !== 'nvidia'`) intact, torch.js delegation intact, no new package install lines
- Phase 1 regression on RTX 4070: `verify_gpu.py` exits 0 (`cuda_available: true`, arch list contains `sm_75` and `sm_86`); `python app.py` serves `Running on local URL: http://127.0.0.1:7860` with `share=False`; `run.js` `input.event[1]` URL capture intact; Standard variant `enhance()` produces a 48,000-frame WAV (`verify_inference.py` exits 0, `ok: true`)

## Task Commits

1. **Task 1: Update pinokio.json metadata + confirm menu.js + regression verify on RTX 4070** - `7314ac1` (feat)

## Files Created/Modified

- `pinokio.json` - launcher metadata: `model-variants` + `realtime` tags added, homepage references Real-Time RE-USE Space
- `app.py` - deviation fix (see below): variant recipe configs load only when present, keeping the Standard Phase 1 path runnable without the missing RealTime recipe

## Decisions Made

- Homepage set to the Real-Time RE-USE Space (variant reference) while retaining the NVIDIA/RE-USE repository link
- No menu.js or install.js edits: variant selection lives in app.py's Gradio dropdown; install stays NVIDIA-only per Phase 1 D-01/BK-02

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing error handling / Rule 3 - Blocking issue] app.py crashed at import because the RealTime recipe YAML is missing**
- **Found during:** Task 1 regression — `python app.py` failed with `FileNotFoundError: recipes/USEMamba_12x1_...hop_160_...early_005_release_random_layer_GAN_longer_1k.yaml` (referenced by `VARIANT_REGISTRY`, added in Plan 02-01, but the recipe file was never created)
- **Issue:** `_VARIANT_FULL_CFG` loaded every variant's recipe unconditionally at import, so one missing recipe file killed the whole app — including the Standard Phase 1 path this plan must re-verify
- **Fix:** Recipe configs now load only when the file exists (a missing variant is logged with a warning and marked unavailable); `enhance()` raises a clear error if an unavailable variant is selected; `cfg1`/`standard` loads unchanged. No model config content was fabricated
- **Files modified:** app.py
- **Verification:** `import app` now succeeds with only `standard` in `_VARIANT_FULL_CFG`; the full regression (app launch, Gradio URL, Standard enhance) passes
- **Committed in:** 7314ac1 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 + Rule 3)
**Impact on plan:** The app.py fix was required to run the plan's own regression (app.py must start); Standard Phase 1 behavior is fully preserved. No scope creep beyond unblocking the regression.

## Issues Encountered

- The RealTime recipe YAML referenced by Plan 02-01's `VARIANT_REGISTRY` does not exist in the repository, so the RealTime variant cannot load at runtime. Handled gracefully (warning at import, clear error on selection) rather than fabricating model config. RealTime end-to-end inference remains pending (it was scoped to Plan 02-01's tracer slice).
- Open item inherited from Phase 1: clean install from scratch not yet re-executed (INST-01 / VER-03 in STATE.md).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02-04 complete; metadata documents RealTime variant support and Phase 1 regression is green on RTX 4070
- Plan 02-05 (final plan in phase) can proceed

---
*Phase: 02-enable-realtime-reuse-variants*
*Completed: 2026-11-12*

## Self-Check: PASSED

- Commit `7314ac199a1f53b8a1105d5d7b057f86b752ded4` is an ancestor of HEAD: yes
- pinokio.json exists and contains model-variants/realtime tags with gpu ["nvidia"]: yes
- app.py exists and imports cleanly (Standard variant loads): yes
