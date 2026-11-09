# Phase 1 Audit: Prior-AI Edits vs Evidence

**Audited:** 2026-11-09
**Scope:** uncommitted working-tree changes vs `HEAD` (`50efcfb`)
**Method:** compare `git diff`, installed packages in `.env`, the pinned mamba source,
and archived logs under `logs/api/`.

## Verdict summary

| File | Change | Evidence | Verdict |
|------|--------|----------|---------|
| `.gitignore` | Added `causal-conv1d/` and `mamba/` | Both are build clones created during install; they must not be committed | **KEEP** |
| `install.js` | Passes `conda: path.resolve(__dirname, '.env')` to `torch.js` | Previous torch run activated `base` and failed: `logs/api/torch.js/1794214330492` shows `error: No virtual environment found` | **KEEP intent, FIX path** |
| `install.js` | Sets `MAMBA_FORCE_BUILD=TRUE` and `MAMBA_KEEP_CUDA_BUILD=FALSE` | Pinned tag `v2.3.2.post1` reads `MAMBA_SKIP_CUDA_BUILD`; `MAMBA_KEEP_CUDA_BUILD` does not exist in that tag (`git show v2.3.2.post1:setup.py`) | **REVERT both** |
| `install.js` | `git clone --branch v2.3.2.post1 --depth 1` | Pinned tag is the proven setup; local clone is nine commits past it (`e9594ce`) | **KEEP** |
| `install.js` | Guard `when: "{{!exists('mamba')}}"` | A partial/failed clone leaves the directory and blocks re-clone: `logs/api/install.js/1794215211886` shows `fatal: destination path 'mamba' already exists` | **FIX to `mamba/setup.py`** |
| `torch.js` | `torchvision==0.25.0` | Installed stack is `torch 2.10.0+cu128` + `torchvision 0.25.0+cu128`; old `0.15.0` does not match torch 2.10 | **KEEP** |
| `torch.js` | Retains AMD Windows/DirectML, AMD ROCm, Apple Silicon, Intel Mac, CPU fallback branches | `pinokio.json` declares `"gpu": ["nvidia"]`; `app.py` uses `device = "cuda"`; README claims no CPU fallback. The dead branches contradict all three | **REVERT / REMOVE** |
| `reset.js` | Deletes `causal-conv1d` and `mamba` | Legacy build folders from earlier installers; `.gitignore` already ignores them | **KEEP as legacy cleanup** |
| `reset.js` | Also deletes `logs` and `output` | `output/` holds the user's enhanced audio; `logs/` is the troubleshooting evidence trail used throughout this audit | **FIX: remove these two deletions** |
| `requirements_full.txt` | Dropped `torchcodec` | `soundfile` is the active reader; `torchcodec.dll` caused failures the app avoids | **KEEP** |
| `README.md` | Added "Measured on a Quadro RTX 6000 (22.5 GB)" | No Quadro RTX 6000 was available; the card has 24 GB, not 22.5 GB; no benchmark log exists | **REVERT** |
| `README.md` | Repeats `MAMBA_KEEP_CUDA_BUILD=FALSE` | Same dead flag as `install.js` | **REVERT** |
| `README.md` | Documents `.mp4`/`.mov` input | `soundfile` cannot demux video; `logs/api/run.js/1794217166729` shows `LibsndfileError` on an `.m4a`/video input | **FIX: audio-only docs** |
| `README_PINOKIO.md` | Repeats stale VRAM and mamba-flag claims | Same evidence as `README.md` | **FIX** |
| `app.py` | Not modified in the working tree; selective-scan patch is committed in `bcd38d1` | Installed site-packages shows `selective_scan_cuda = None` and the source guard | **KEEP, extend in this phase** |

## GPU conclusion

**The prior edits added no new GPU architecture.** They improved install robustness and
environment correctness. GPU coverage was already present in the CUDA 12.8 wheel.

Evidence (installed `.env`, RTX 4070 machine):

```
torch 2.10.0+cu128
torchvision 0.25.0+cu128
cuda True
arch ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
dev NVIDIA GeForce RTX 4070
cap (8, 9)
```

- `sm_75` is the Turing target for Quadro RTX 6000. Coverage is **reasoned** from the
  wheel architecture list, never executed on that card.
- `sm_86` is the binary-compatible path for the RTX 4070 (compute 8.9 runs 8.6 SASS).
- No new architecture was introduced by the uncommitted edits; the same wheel already
  covered both cards before and after the changes.

## Environment findings

- The installed `.env` is a conda environment with Python 3.10.22.
- `mamba_ssm 2.3.2.post1` imports successfully.
- `mamba_ssm.ops.selective_scan_interface.selective_scan_cuda is None` after the
  runtime patch; `selective_scan_ref` is callable.
- `causal_conv1d` is installed but broken:
  `ModuleNotFoundError: No module named 'causal_conv1d_cuda'`.
- `mamba_ssm.modules.mamba_simple` imports `causal_conv1d` inside `try/except ImportError`,
  so the broken package does not block the app. `causal-conv1d` is **optional** and must
  not be installed by the launcher.
- The pinned tag `v2.3.2.post1` uses `MAMBA_SKIP_CUDA_BUILD`; the local `mamba/`
  checkout at `e9594ce` uses `MAMBA_KEEP_CUDA_BUILD`. Mixing the two is the source of the
  dead-flag confusion. This phase pins the tag deliberately.

## Evidence baseline

| Fact | Value | Source |
|------|-------|--------|
| GPU | NVIDIA GeForce RTX 4070, 12 GB, compute (8, 9) | `.env` probe |
| Torch | `2.10.0+cu128` | `.env` probe |
| Torchvision | `0.25.0+cu128` | `.env` probe |
| Torchaudio | `2.10.0` | `.env` probe |
| Arch list | `['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` | `.env` probe |
| Mamba | `2.3.2.post1`, `selective_scan_cuda = None` | `.env` probe |
| Broken optional | `causal_conv1d_cuda` missing | `.env` probe |
| Successful inference | 72.9 s on RTX 4070 | `logs/api/run.js/1794218407913` |
| Gradio share failure | `Could not create share link` + `h11` error | `logs/api/run.js/1794218407913` |
| Torch install failure | `error: No virtual environment found` | `logs/api/torch.js/1794214330492` |
| Mamba clone failure | `fatal: destination path 'mamba' already exists` | `logs/api/install.js/1794215211886` |
| Audio read failure | `LibsndfileError` on `.m4a`/video | `logs/api/run.js/1794217166729` |
| Requirements install | Succeeded in `.env`, Python 3.10.22 | `logs/api/install.js/1794214330416` |

## Decisions carried into the phase

- Keep the conda `.env` with Python 3.10; do not switch to a plain venv.
- Use relative paths in launcher scripts.
- Pin the torch family as `torch 2.10.0 / torchvision 0.25.0 / torchaudio 2.10.0`.
- Build mamba with `MAMBA_SKIP_CUDA_BUILD=TRUE` only.
- Do not install `causal-conv1d`.
- Remove all non-NVIDIA torch branches.
- Fail install with a clear message when no NVIDIA GPU is present.
- Remove the fabricated Quadro table and the video input claim.
- Add `verify_gpu.py` + `verify.js` and wire them into install and the menu.
