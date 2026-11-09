---
phase: 01-nvidia-install-and-gpu-truth
verified: 2026-11-09T15:42:09Z
status: human_needed
score: 20/20 must-haves verified
behavior_unverified: 1
behavior_unverified_items:
  - truth: "A clean install on Windows + RTX 4070 completes from scratch with no manual steps"
    test: "Delete .env, cache, and mamba, then run install.js through Pinokio"
    expected: "install.js reaches the final 'Installation verified' step and verify_gpu.py exits 0"
---

# Phase 1 Verification Report

**Phase:** 01-nvidia-install-and-gpu-truth
**Verified:** 2026-11-09
**Host:** Windows x64, NVIDIA GeForce RTX 4070 (12 GB, compute 8.9)
**Method:** executed commands inside the launcher's `.env`, static inspection of every
launcher script, and the archived logs cited in `01-AUDIT.md`.

## Verdict

The launcher is now NVIDIA-only, fails fast on unsupported machines, and ships a single
repeatable verifier that passes on the RTX 4070. The app loads the model, enhances real
audio, and exposes a local Gradio URL that matches `run.js`'s capture pattern.

One roadmap success criterion is **not** closed: the new `install.js` has not been run
from a truly clean state on this machine. The existing `.env` was reused to avoid
destroying a working environment. That is recorded below as the only open behavior item.
Quadro RTX 6000 support remains **reasoned**, never measured.

**Status:** `human_needed` — run `Install` once from a clean state to close INST-01.

## Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every prior-AI edit is classified keep/fix/revert with evidence | ✓ verified | `01-AUDIT.md` verdict table cites `logs/api/...` and `git show v2.3.2.post1:setup.py` |
| 2 | Audit states the prior edits added no new GPU architecture | ✓ verified | `01-AUDIT.md` "GPU conclusion" section |
| 3 | Audit names the dead mamba flag and the broken `causal-conv1d` state | ✓ verified | `01-AUDIT.md` "Environment findings" |
| 4 | Install creates `.env` with Python 3.10 declaratively | ✓ verified (static) | `install.js` uses `conda: { path: '.env', python: 'python=3.10' }`; `.env` reports 3.10.22 |
| 5 | Install reuses `.env` on re-run | ✓ verified (static) | conda path activation step plus `!exists('mamba/setup.py')` clone guards |
| 6 | Pinned mamba tag builds with `MAMBA_SKIP_CUDA_BUILD` only | ✓ verified (static) | `install.js` sets only that variable; tag source was read in the audit |
| 7 | Torch family pins are mutually compatible | ✓ verified | `verify_gpu.py`: `2.10.0+cu128` / `0.25.0+cu128` / `2.10.0+cu128` |
| 8 | Reset removes only install artifacts | ✓ verified (static) | `reset.js` removes `.env`, `cache`, `mamba`, legacy `causal-conv1d`; keeps `output/` and `logs/` |
| 9 | `torch.js` is NVIDIA cu128 only and fails fast elsewhere | ✓ verified | two cu128 branches plus a `break: true` guard; `require('./torch.js')` succeeds |
| 10 | `verify_gpu.py` checks CUDA, device, capability, arch, torch family, mamba | ✓ verified | JSON output below |
| 11 | `app.py` refuses to run without CUDA | ✓ verified | `raise SystemExit(...)` guarded by `torch.cuda.is_available()` |
| 12 | App launches locally without a share tunnel | ✓ verified | log shows `Running on local URL: http://127.0.0.1:7860` and the share hint only |
| 13 | Video tab removed; UI is audio-only | ✓ verified | `app.py` has no `Video Upload`; single `gr.Audio` input |
| 14 | `menu.js` exposes Verify GPU and `run.js` keeps its URL capture | ✓ verified | `menu.js` contains `Verify GPU` → `verify.js`; `run.js` still has `input.event[1]` |
| 15 | README has no fabricated Quadro measurement or wrong capacity | ✓ verified | no `22.5`; compatibility table uses Tested / Reasoned / Unsupported |
| 16 | README records what was learned and unlearned | ✓ verified | "What we learned / What we unlearned" section |
| 17 | README documents supported OS, GPU, Python, disk, audio formats | ✓ verified | Requirements + Supported configurations sections |
| 18 | Install and run instructions match the scripts | ✓ verified | README mirrors `install.js`, `torch.js`, `run.js`, `menu.js` |
| 19 | A repeatable command proves the GPU stack | ✓ verified | `python verify_gpu.py` exits 0 |
| 20 | A verification report records evidence and unknowns | ✓ verified | this document |

## Required Artifacts

| Artifact | Expected | Status |
|----------|----------|--------|
| `.planning/.../01-AUDIT.md` | Change-by-change audit | ✓ exists, `MAMBA_SKIP_CUDA_BUILD` + "no new GPU architecture" present |
| `install.js` | Declarative conda, cu128, mamba gate, final verifier | ✓ 11 steps, `requires.bundle = "ai"` |
| `reset.js` | Artifact-only cleanup | ✓ 5 steps, no `logs`/`output` |
| `torch.js` | NVIDIA-only cu128 | ✓ two cu128 branches, fail-fast guard |
| `reuse_runtime.py` | Single mamba patch | ✓ exports both helpers |
| `verify_gpu.py` | JSON verifier, non-zero on failure | ✓ `sys.exit(1)` path present |
| `verify.js` | Pinokio entry point | ✓ runs `python verify_gpu.py` in `.env` |
| `verify_inference.py` | End-to-end smoke test | ✓ uses `default_rng(0)`, calls `app.enhance` |
| `menu.js` | Verify GPU entry | ✓ present in installed branch |
| `app.py` | CUDA guard, audio-only, `share=False` | ✓ `if __name__ == "__main__":` present |
| `README.md` / `README_PINOKIO.md` | Honest docs | ✓ rewritten |

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `install.js` | `torch.js` | `script.start` with `conda: '.env'` | ✓ |
| `install.js` | `verify_gpu.py` | final `python verify_gpu.py` | ✓ |
| `torch.js` | cu128 wheels | `--index-url https://download.pytorch.org/whl/cu128` | ✓ (two occurrences) |
| `verify.js` | `verify_gpu.py` | `shell.run` inside `.env` | ✓ |
| `menu.js` | `verify.js` | menu item href | ✓ |
| `app.py` | `reuse_runtime.py` | `patch_mamba_selective_scan` + `bind_selective_scan_reference` | ✓ |
| `run.js` | Gradio URL | `input.event[1]` from `/(http:\/\/[0-9.:]+)/` | ✓ unchanged; URL format matches |

## Executed Evidence

### `python verify_gpu.py`

```
{
  "ok": true,
  "torch": "2.10.0+cu128",
  "torchvision": "0.25.0+cu128",
  "torchaudio": "2.10.0+cu128",
  "cuda_available": true,
  "device": "NVIDIA GeForce RTX 4070",
  "capability": [
    8,
    9
  ],
  "arch_list": [
    "sm_70",
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_90",
    "sm_100",
    "sm_120"
  ],
  "required_arch": [
    "sm_75",
    "sm_86"
  ],
  "mamba_ssm": "2.3.2.post1",
  "mamba_patch_applied": true,
  "selective_scan_cuda": null,
  "selective_scan_ref": true,
  "selective_scan_bound": true,
  "errors": []
}
```

### `python verify_inference.py`

Command environment: `HF_HOME=cache/HF_HOME`, `TORCH_HOME=cache/TORCH_HOME`,
`GRADIO_TEMP_DIR=cache/GRADIO_TEMP_DIR`, `MPLBACKEND=Agg`.

```
Loading model (first time, may take a while)...
Model loaded.
[Timer] Inferencia oficial por chunks completada: 4.2777s
{
  "ok": true,
  "device": "NVIDIA GeForce RTX 4070",
  "input_sr": 16000,
  "output_path": "output\\verify_input_enhanced.wav",
  "output_frames": 48000,
  "elapsed_seconds": 20.343,
  "errors": [],
  "output_sr": 16000
}
```

`elapsed_seconds` includes the one-time model download/load. The chunk timer for the
3-second clip was 4.28 s.

### Gradio local launch

`python app.py` with the same cache env, `PYTHONUNBUFFERED=1`, `share=False`:

```
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

The URL was captured 8.1 s after process start. There is no share-link attempt and no
`h11` error. The URL matches `run.js`'s `/(http:\/\/[0-9.:]+)/` pattern.

## Requirements Coverage

| Requirement | Status | Evidence / note |
|-------------|--------|-----------------|
| INST-01 | Unverified | `install.js` was not executed from a clean state; see open item |
| INST-02 | Reasoned | Re-run safety comes from `!exists('mamba/setup.py')`, the incomplete-clone cleanup, and conda reuse; not re-executed |
| INST-03 | Verified | `verify_gpu.py` imports torch/torchvision/torchaudio at matching cu128 versions |
| INST-04 | Verified | `install.js` sets only `MAMBA_SKIP_CUDA_BUILD`; the pinned tag reads exactly that |
| INST-05 | Verified | Declarative `python=3.10`; `.env` is 3.10.22 |
| INST-06 | Verified (static) | `reset.js` removes only `.env`, `cache`, `mamba`, `causal-conv1d` |
| GPU-01 | Verified | Installed wheel is `2.10.0+cu128` from the cu128 index on Windows; Linux branch not executed |
| GPU-02 | Verified | `cuda_available: true`, arch list contains `sm_75` and `sm_86` |
| GPU-03 | Reasoned | Guard exists in `install.js` and `torch.js`; no non-NVIDIA machine was available to execute it |
| GPU-04 | Verified | `pinokio.json` declares `nvidia`; README names unsupported platforms |
| GPU-05 | Verified (doc) | Quadro RTX 6000 documented as reasoned from `sm_75`, never as tested |
| REL-01 | Verified | `verify_inference.py` ran the real model on the RTX 4070 |
| REL-02 | Verified | `mamba_ssm 2.3.2.post1` imports; `selective_scan_ref` callable and bound |
| REL-03 | Verified | Broken `causal_conv1d_cuda` is present but optional; the new installer does not install it |
| REL-04 | Reasoned | `run.js` is unchanged and its pattern matches the captured URL; the Pinokio daemon path was not re-run |
| DOC-01 | Verified | No `22.5` and no fabricated Quadro table |
| DOC-02 | Verified | Learned/unlearned section covers arch coverage, torchvision, mamba flag, `.m4a` |
| DOC-03 | Verified | README requirements list OS, GPU, driver, audio formats, disk |
| DOC-04 | Verified | Instructions mirror the scripts |
| VER-01 | Verified | `python verify_gpu.py` |
| VER-02 | Verified | This report |
| VER-03 | Unverified | Start and inference were exercised, but not as a clean-install → start → inference sequence |

## Unverified / Reasoned

- **Quadro RTX 6000 compatibility**: reasoned from the installed wheel's `sm_75`
  entry. No Quadro RTX 6000 was available; no inference or benchmark was run on it.
- **Linux install path**: the branch exists in `torch.js` and `pinokio.json`, but only
  Windows was executed on this machine.
- **No-NVIDIA fail-fast path**: the guard is present and syntactically valid, but no
  AMD/CPU machine was available to observe the error screen.
- **Clean install from scratch**: the new `install.js` has not been executed from an
  empty `.env`. This is the one open behavior item (`behavior_unverified: 1`).
- **Reset behavior**: `reset.js` content was inspected, not executed, to avoid deleting
  the working environment.
- **Long-clip / high-sample-rate VRAM behavior**: no benchmark exists. The README now
  states that explicitly instead of publishing invented numbers.

## Commands Used

```powershell
# GPU and runtime verification
.\.env\python.exe verify_gpu.py

# End-to-end inference
$env:HF_HOME="cache\HF_HOME"; $env:TORCH_HOME="cache\TORCH_HOME"
$env:GRADIO_TEMP_DIR="cache\GRADIO_TEMP_DIR"; $env:MPLBACKEND="Agg"
.\.env\python.exe verify_inference.py

# Local Gradio launch (captured, then process tree terminated)
$env:PYTHONUNBUFFERED="1"
.\.env\python.exe app.py
```

## Next Step to Close the Phase

Run **Install** once from a clean state:

1. `pinokio` → app menu → **Reset** (removes `.env`, `cache`, `mamba`; keeps `output/` and `logs/`).
2. **Install** and confirm it reaches "Installation verified".
3. **Run RE-USE** and confirm the captured local URL opens.

After that, INST-01 and VER-03 can move from Unverified to Verified and the phase can be
marked `passed`.
