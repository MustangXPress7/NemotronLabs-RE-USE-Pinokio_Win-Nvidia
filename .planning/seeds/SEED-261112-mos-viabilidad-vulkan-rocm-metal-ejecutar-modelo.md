---
id: SEED-261112-mos
status: dormant
planted: 2026-11-12T00:00:00Z
planted_during: Phase 1: NVIDIA install and GPU truth
trigger_when: when relevant
scope: unknown
---

# SEED-261112-mos: viabilidad de usar Vulkan, ROCm, Metal o OpenVINO para ejecutar este modelo

## Why This Matters

_To be filled in. Run `/gsd-capture --seed --enrich SEED-261112-mos` to add context._

## When to Surface

**Trigger:** when relevant

This seed will surface during `/gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `/gsd-capture --seed --enrich SEED-261112-mos` to estimate effort.

## Breadcrumbs

- `install.js:9` — `when: "{{gpu !== 'nvidia' || (platform !== 'win32' && platform !== 'linux')}}"` fail-fast sin GPU NVIDIA
- `install.js:56-59` — torch.js con `triton: true`, xformers/sageattention/flashattention false
- `app.py:39-43` — `torch.cuda.is_available()` raise SystemExit si no CUDA
- `pinokio.json:7-8` — `"arch": ["x64"]`, `"gpu": ["nvidia"]`
- `pinokio.json:6` — `"platform": ["win32", "linux"]`
- `reuse_runtime.py` — parche de mamba_selective_scan (CUDA-only path)
- `pinokio.json:6-8` — ninguna referencia a OpenVINO, Intel, Vulkan, ROCm o Metal; solo NVIDIA/CUDA

## Notes

_Captured via one-shot seed capture. Enrich with trigger, why, and scope at your convenience._

Idea captured: evaluar si es viable portar/adaptar el modelo RE-USE para ejecutarse sobre backends no-CUDA: Vulkan (multiplataforma GPU), ROCm (AMD, Linux), Metal (Apple Silicon) o OpenVINO (Intel CPU/iGPU/dGPU), rompiendo el acoplamiento actual NVIDIA-only.
