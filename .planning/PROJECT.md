# RE-USE Pinokio Launcher

## What This Is

Pinokio launcher for NVIDIA's RE-USE universal speech enhancement model. It installs a
Python environment, installs the PyTorch/Mamba/Triton stack needed to run the model on
NVIDIA GPUs, and starts the Gradio interface locally so a user can clean audio files with
one click.

## Core Value

A clean install on a supported NVIDIA machine produces a working GPU inference environment
and a running Gradio app, without hidden manual steps and without documented claims that
were never verified.

## Requirements

### Validated

- ✓ Launcher installs Python dependencies into a local `.env` virtual environment — existing launcher
- ✓ Launcher starts the RE-USE Gradio app and surfaces the local URL — existing launcher
- ✓ Model loads and performs inference on an RTX 4070 with CUDA 12.8 PyTorch — observed in `logs/api/run.js/1794218407913`

### Active

- [ ] Install flow is reproducible from a clean state on Windows/NVIDIA
- [ ] CUDA 12.8 GPU support is explicit and truthful for NVIDIA only
- [ ] Install flags and pinned versions actually affect the build they claim to affect
- [ ] No regression on RTX 4070: the app still loads the model and enhances audio on GPU
- [ ] Documentation states only verified facts and records what was learned and unlearned
- [ ] A verification artifact proves GPU support and clean-install behavior

### Out of Scope

- AMD ROCm support — no ROCm hardware available and Mamba/Triton support is not proven; promising it would be dishonest
- Apple Silicon / MPS support — the Mamba CUDA kernels and the app's `device = "cuda"` design are NVIDIA-specific
- CPU-only inference — the app hardcodes CUDA; a CPU fallback is an app redesign, not a launcher fix
- Real validation on Quadro RTX 6000 — hardware not available; compatibility is reasoned from wheels and `sm_75`, never executed
- Docker-based install — Pinokio manages native per-platform installs

## Context

This is a brownfield launcher around an existing research app. The app lives at the
repository root (not in an `app/` subfolder), with launcher scripts (`install.js`,
`torch.js`, `start.js`, `reset.js`, `pinokio.js`) alongside it.

Prior work by another AI changed `install.js`, `torch.js`, `reset.js`, README files,
`.gitignore` and `requirements_full.txt`. Those changes are **uncommitted and untested**;
the only successful run evidence predates them.

Verified environment facts:

- GPU used for validation: NVIDIA RTX 4070, 12 GB, compute capability 8.9.
- Installed PyTorch: `2.10.0+cu128` with arch list `sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`.
- `sm_75` covers Quadro RTX 6000 (Turing); `sm_86` covers RTX 4070 (Ada, binary compatible).
- `torchvision 0.25.0+cu128` matches torch 2.10; the previously pinned `0.15.0` was wrong.
- The pinned `mamba` tag `v2.3.2.post1` hard-imports `selective_scan_cuda`; `app.py` monkey-patches site-packages at runtime to use a pure-Python selective scan.
- `MAMBA_KEEP_CUDA_BUILD` is ignored by that tag; it uses `MAMBA_SKIP_CUDA_BUILD`.
- `causal-conv1d` is installed but broken (`ModuleNotFoundError: causal_conv1d_cuda`) and is not required by the working run.
- The original `torch.js` had a Windows CUDA fallback that has been removed in the working tree.
- Original install evidence shows torch install failed with "No virtual environment found" and the mamba clone/build failed.
- `soundfile` cannot read `.m4a`/video files; audio must be converted first.

## Constraints

- **Platform**: Windows and Linux, x64 only — declared in `pinokio.json`.
- **GPU**: NVIDIA only — no AMD/Apple promise until hardware and kernels exist.
- **Hardware validation**: RTX 4070 available; Quadro RTX 6000 must be reasoned, not executed.
- **Pinokio**: scripts must follow Pinokio API patterns from `D:\pinokio\prototype\system\examples`; no ad-hoc absolute paths.
- **App code**: avoid modifying the upstream app unless required; prefer launcher-side fixes.
- **Working tree**: contains uncommitted prior-AI edits that must be reviewed, not blindly trusted.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| NVIDIA-only GPU scope | Only NVIDIA hardware and CUDA kernels are available and proven | — Pending |
| Validate on RTX 4070 only | Quadro RTX 6000 is not accessible; reasoning from `sm_75` is the honest limit | — Pending |
| Single GSD phase producing plan + implementation + verification | The work is tightly coupled; splitting would leave unverified scripts between phases | — Pending |
| Keep one GSD phase with multiple plans | Matches the requested structure and keeps a single coherent deliverable | — Pending |
| Documentation must distinguish verified from reasoned claims | Prior README contained unverified Quadro measurements | — Pending |

---

*Last updated: 2026-11-09 after GSD bootstrap*
