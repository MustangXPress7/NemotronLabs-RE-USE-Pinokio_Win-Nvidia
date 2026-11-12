---
phase: 02-enable-realtime-reuse-variants
plan: 05b
subsystem: ui
tags: [gradio, mamba, speech-enhancement, nvidia, cuda, model-variants, realtime, laughter, whisper, bf16]

# Dependency graph
requires:
  - phase: 01-nvidia-install-and-gpu-truth
    provides: Phase 1 working app.py with SEMamba from_pretrained Hub loading, device="cuda" guard, share=False launch, mamba selective-scan patch
  - plan: 02-01
    provides: VARIANT_REGISTRY dict, Model Variant dropdown, variant-aware model loading
provides:
  - 3 new model variants: Laughter, Whisper, BF16
  - 3 new recipe YAMLs in recipes/
  - 5-variant Gradio dropdown UI
  - Variant-dependent slider enable/disable (RealTime + Whisper)
  - All 5 variants verified on RTX 4070 via app.enhance()
affects: [02-01, 02-02, 02-03, 02-04, 02-05]

# Actuals (#2632)
actuals:
  tokens: 2500
  tasks: 1
  commits: 1

# Tech tracking
tech-stack:
  added: []
  patterns: [variant registry expansion, 5-variant Gradio dropdown, variant-dependent UI slider enable/disable]

key-files:
  created:
    - recipes/USEMamba_30x1_laughter_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml
    - recipes/USEMamba_12x1_whisper_lr_00002_norm_05_vq_067_nfft_320_hop_160_NRIR_012_pha_0005_com_04_early_005_release_random_layer_GAN_longer_1k.yaml
    - recipes/USEMamba_30x1_bf16_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml
  modified:
    - app.py - 5-variant VARIANT_REGISTRY, 5-choice dropdown, Whisper slider enable/disable

key-decisions:
  - "Laughter variant uses same architecture as Standard (30-block SEMamba, hop_40), different weights only"
  - "Whisper variant uses same architecture as RealTime (12-block SEMamba_decoder_list, hop_160), fine-tuned for whispered speech"
  - "BF16 variant uses same architecture as Standard, BF16 storage for smaller disk footprint (21.1 MB vs 38.6 MB)"
  - "Whisper shares exit_layer/look_ahead_frames sliders with RealTime (variant-dependent enable/disable)"
  - "All variants load via from_pretrained from Hugging Face Hub (no local .pth files needed)"

requirements-completed: [RTV-04, RTV-05, RTV-06]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: Laughter variant (nvidia/NVIDIA-NemotronLabs-RE-USE-laughter) loads and infers on RTX 4070
    requirement: RTV-05
    verification:
      - kind: integration
        ref: app.enhance('test_variants.wav', variant_id='laughter') → 16,000-frame WAV
        status: pass
      - kind: manual_procedural
        ref: app.py syntax check + variant dropdown shows 5 choices
        status: pass
    human_judgment: true
    rationale: Real-time audio quality assessment requires human listening; automated SI-SDR not run
  - id: D2
    description: Whisper variant (Malfaro43/Realtime-REUSE-whisper-finetune) loads and infers on RTX 4070
    requirement: RTV-04
    verification:
      - kind: integration
        ref: app.enhance('test_variants.wav', variant_id='whisper') → 16,000-frame WAV
        status: pass
      - kind: manual_procedural
        ref: app.py syntax check + whisper slider enable/disable works
        status: pass
    human_judgment: true
    rationale: Real-time audio quality assessment requires human listening; automated SI-SDR not run
  - id: D3
    description: BF16 variant (giangndm/NVIDIA-NemotronLabs-RE-USE-bf16) loads and infers on RTX 4070
    requirement: RTV-06
    verification:
      - kind: integration
        ref: app.enhance('test_variants.wav', variant_id='bf16') → 16,000-frame WAV
        status: pass
      - kind: manual_procedural
        ref: app.py syntax check + BF16 model loads without error
        status: pass
    human_judgment: true
    rationale: Real-time audio quality assessment requires human listening; automated SI-SDR not run

# Metrics
duration: 45min
completed: 2026-11-12
status: complete

# Phase 02 Plan 05b: 3 New Model Variants Summary

**5 model variants now supported in the RE-USE Pinokio launcher: Standard, RealTime, Laughter, Whisper, BF16**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-11-12
- **Completed:** 2026-11-12
- **Tasks:** 1
- **Files modified:** 4 (app.py + 3 recipe YAMLs)

## Accomplishments

- Added Laughter variant (nvidia/NVIDIA-NemotronLabs-RE-USE-laughter): 9.6M params, 30-block SEMamba, preserves laughter better
- Added Whisper variant (Malfaro43/Realtime-REUSE-whisper-finetune): 13.5M params, 12-block SEMamba_decoder_list, fine-tuned for whispered speech
- Added BF16 variant (giangndm/NVIDIA-NemotronLabs-RE-USE-bf16): 9.6M params, 30-block SEMamba, BF16 storage, smaller disk footprint (21.1 MB)
- Gradio dropdown now shows 5 variants: Standard, RealTime, Laughter, Whisper, BF16
- Whisper shares exit_layer/look_ahead_frames sliders with RealTime (variant-dependent enable/disable)
- All 3 new variants verified on RTX 4070 via app.enhance() — 16,000-frame WAV output each

## Task Commits

Each task was committed atomically:

1. **Add 3 model variants + dropdown expansion + recipe YAMLs** - `f79e9f8` (feat)

## Files Created/Modified

- `app.py` - 5-variant VARIANT_REGISTRY, 5-choice Gradio dropdown, Whisper slider enable/disable
- `recipes/USEMamba_30x1_laughter_*.yaml` - Laughter recipe YAML (downloaded from HF Hub)
- `recipes/USEMamba_12x1_whisper_*.yaml` - Whisper recipe YAML (downloaded from HF Hub)
- `recipes/USEMamba_30x1_bf16_*.yaml` - BF16 recipe YAML (downloaded from HF Hub)

## Decisions Made

- All variants use `weight_source: "hf_hub"` — weights download from Hugging Face Hub automatically
- Whisper shares RealTime's exit_layer/look_ahead_frames sliders (same 12-block architecture)
- Laughter and BF16 share Standard's chunked inference (same 30-block architecture)
- No new Python packages required — Phase 1 stack (torch 2.10.0+cu128, mamba-ssm 2.3.2.post1) covers all variants

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all variants fully functional.

## Threat Flags

None - no new security-relevant surface introduced.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 5 variants ready for user testing via Gradio UI
- MLX/LiteRT models documented for future Mac version (see MLX_MODELS_FOR_MAC.md)
- Phase 2 fully complete with 15/15 requirements verified

---
*Phase: 02-enable-realtime-reuse-variants*
*Completed: 2026-11-12*