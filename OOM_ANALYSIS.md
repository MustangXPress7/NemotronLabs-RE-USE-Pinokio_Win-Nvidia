# Why the RealTime variants run out of memory

This note explains where the memory goes when the **RealTime** variants are used
in this app, why the **Standard** variants do not hit the same wall, and what the
upstream papers and model cards say about it. Every number below was measured on
this machine (RTX 4070, 12 GB) with the instrumentation added in
`reuse_profiler.py`; the raw runs are in `logs/inference/`.

## TL;DR

1. The app runs **one forward pass over the whole file** for the RealTime
   family (`VARIANT_REGISTRY[...]["chunked"] = False` in `app.py`). The model
   was trained and published as a **streaming** model that consumes one 20 ms
   frame at a time; running it offline keeps a workspace proportional to
   `duration x frequency bins` instead of `O(1)`.
2. `mamba_ssm` is installed **without** the compiled `selective_scan_cuda`
   extension (see `reuse_runtime.py`), so every scan goes through
   `selective_scan_ref`, a pure-Python kernel that materialises two float32
   tensors of shape `[batch, d_inner, seq, d_state]` per call.
3. That product is what explodes:

   ```
   peak scan workspace ~= 2 tensors * 4 B * t_frames * f_enc * d_inner * d_state
   total peak          ~= 1.75 * peak scan workspace
   ```

   `t_frames` grows with duration, `f_enc` grows with the **input sample rate**
   (48 kHz -> 481 bins, 16 kHz -> 161, 8 kHz -> 81), and `d_inner = 256`,
   `d_state = 16` are fixed by the recipe.
4. Measured cost at 48 kHz mono: **~1.26 GB per second of audio** (~6.3 GB for a
   5 s clip). A 12 GB card is out of room at roughly **8-9 s** of 48 kHz audio.
   At 16 kHz it is ~0.43 GB/s, at 8 kHz ~0.22 GB/s.
5. On Windows (WDDM) exceeding VRAM does not always raise `CUDA out of memory`:
   the driver silently spills into shared system memory. A 12 s / 48 kHz clip
   peaked at **15.04 GB** on an 11.99 GB card and took **240 s** (RTF 20)
   instead of failing. If a RealTime run "hangs", check the peak in the log.

## Where the memory goes

### 1. The app chooses single-pass for RealTime

```python
"realtime": { ..., "chunked": False, "family": "realtime" }
```

`enhance()` therefore takes this branch:

```python
Noisy_wav = Noisy_wav.to(device)          # whole file on the GPU
noisy_mag, noisy_pha, _ = mag_phase_stft(Noisy_wav, ...)   # whole file STFT
amp_g, pha_g, _ = SE_model(noisy_mag, noisy_pha, ...)      # one forward pass
```

No chunking, no overlap-add, and (before this change) no `empty_cache()` and no
memory preflight. The Standard family instead loops over 3 s chunks with 50%
overlap, so its peak is set by the chunk size and not by the file length.

### 2. The reference selective-scan kernel materialises the full state tensor

`mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref` does:

```python
deltaA   = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))     # [b, d, l, n]
deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)        # [b, d, l, n]
```

Both are float32. With `d = d_inner = hid_feature * expand = 64 * 4 = 256` and
`n = d_state = 16`, a single call costs `2 * 4 * b * l * 4096` bytes.

For a RealTime forward pass, `TFMambaBlock.forward` reshapes the coded
spectrogram twice:

```python
x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)   # temporal Mamba
x = x.view(b, f, t, c).permute(0, 2, 1, 3).view(b * t, f, c)  # frequency Mamba
```

* temporal Mamba: `batch = f_enc`, `seq = t_frames`
* frequency Mamba: `batch = t_frames`, `seq = f_enc` (forward + backward, two
  calls, sequential)

All of them share the same `batch * seq = t_frames * f_enc`, which is why a
single formula covers the whole block. Blocks run one after another, so the
peak is one block pass, not the sum over blocks.

### 3. `f_enc` and `t_frames` from the scaled STFT

`app.py` scales the STFT to the input sample rate so the model keeps its time
resolution:

```
n_fft  = make_even(320 * sr / 8000)
hop    = make_even(160 * sr / 8000)
bins   = n_fft / 2 + 1
f_enc  = (bins - 1) / 2 + 1        # DenseEncoder (1,3) conv with stride (1,2)
T      = samples / hop + 1         # center=True
```

Consequences:

| input | hop | frames/s | bins | f_enc | T/f_enc per second |
|-------|-----|----------|------|-------|--------------------|
| 8 kHz | 160 | 50 | 161 | 81 | 50 x 81 |
| 16 kHz | 320 | 50 | 321 | 161 | 50 x 161 |
| 48 kHz | 960 | 50 | 961 | 481 | 50 x 481 |

The frame rate is constant at 50 fps (RealTime) and 200 fps (Standard), but the
frequency axis scales linearly with the sample rate. **Uploading 48 kHz audio
costs 5.9x more scan memory than 8 kHz for the same duration.**

### 4. Measured peak

Realtime, mono, `exit_layer=8`, `look_ahead=0`, 5 s clips, RTX 4070:

| input | T x F | predicted scan | predicted peak | measured peak | model time |
|-------|-------|----------------|----------------|---------------|------------|
| 8 kHz | 251 x 81 | 0.62 GB | 1.09 GB | **1.12 GB** | 0.79 s |
| 16 kHz | 251 x 161 | 1.23 GB | 2.16 GB | **2.16 GB** | 1.54 s |
| 48 kHz | 251 x 481 | 3.68 GB | 6.44 GB | **6.31 GB** | 5.01 s |

The calibrated estimate (`predict_peak_mb`) is within 3% of the measurement.
Extrapolating linearly: 48 kHz mono OOMs around 8-9 s on a 12 GB card, 16 kHz
around 25 s, 8 kHz around 50 s. Stereo doubles the scan workspace because the
encoder keeps channels in the batch dimension (measured: 1 s / 16 kHz / stereo
peaked at 0.94 GB, i.e. ~2x the mono figure for 1 s).

## Secondary contributors

| Factor | Effect | Status |
|--------|--------|--------|
| Pure-Python scan kernel | 5-50x more allocations than the fused kernel; also ~2.5x slower | Install-time decision (`reuse_runtime.py`) |
| `mag_phase_stft` builds `com` | `[B, F, T, 2]` float32 kept alive; unused by the model | `del noisy_com` added for the single-pass path |
| Stereo input | Doubles the scan workspace (batch) | Warned in the log |
| Sample rate > 48 kHz | Grows `f_enc` further | Warned in the log |
| Model cache (`_USE_MODELS`) | Keeps every variant ever loaded resident (~51 MB each) | Intentional; cheap |
| Allocator fragmentation | The scan allocates/frees multi-GB blocks per layer | `empty_cache()` after the STFT and in the chunk loop |
| WDDM shared-memory spill | No OOM, 10-20x slower instead | Reported as "predicted peak > free" warning |
| Exit layer | 3-12 blocks run; peak per block is constant | Lower layers = faster, same peak |

## What the papers and model cards say

**RE-USE** (arXiv 2603.02641, NeurIPS 2026) introduces the SFI-STFT front end and
a 30-layer **bidirectional** (non-causal) Mamba stack for universal speech
enhancement across sample rates, degradations and languages. Being bidirectional,
it cannot stream: every layer sees the whole utterance. The model card tells
users to run `inference_chunk.sh` for audio longer than ~30 s because the demo
otherwise OOMs, which is the same single-pass problem this app reproduces.

**Real-time RE-USE** (arXiv 2606.25621, IEEE SLT 2026) adds the causal variant
used by the `realtime`/`whisper` entries here:

* fixed algorithmic latency of `40 ms + look_ahead x 20 ms`, with look-ahead
  0-2 frames, and an explicit requirement of `RTF < 1` for live use;
* an **early-exit** head: one of 12 layers can be selected (`exit_layer`
  3-12 in this app), each with its own magnitude and phase decoder;
* a strictly causal encoder/decoder pair plus per-layer state caches, so the
  streaming reference implementation (`online_inference.py`) runs with
  `StreamingCausalConv2d`, `allocate_inference_cache()` and `mamba.step()`:
  memory per frame is **O(1)**, independent of how long the session runs;
* the official `offline_inference.py` (the batch script this app mirrors)
  evaluates the full utterance in parallel and the paper warns that measuring
  RTF this way "can substantially underestimate the true RTF".

So the OOM is not a property of the model; it is the cost of evaluating a
streaming architecture in offline, parallel-over-the-whole-file mode. The
published 12-layer configuration is already tight against its 20 ms hop
(computational latency ~25 ms on an A100, RTF ~1.25), which is why the pad-free
`look_ahead` and `exit_layer` knobs exist at all.

## Mitigations, in order of effectiveness

| Mitigation | Effect | Cost |
|------------|--------|------|
| Use **Standard** (chunked) for long files | Peak set by chunk size (1.37 GB for 3 s / 16 kHz stereo) | Not the low-latency model |
| Split the file into 5-8 s clips | Linear reduction | Manual work; chunk joins |
| Mono | ~2x less memory | Channel loss |
| Resample to 16 kHz / 8 kHz | ~3x / ~6x less memory at 48 kHz | Bandwidth above 8/16 kHz lost |
| Lower `exit_layer` | Faster, less fragmentation; peak unchanged | Quality |
| `torch.cuda.empty_cache()` between chunks | Avoids fragmentation | Small overhead |

A true fix for RealTime would be to port the streaming path from
`online_inference.py`: cache the causal convolutions, allocate the Mamba states
once, and feed one frame per step. That keeps all RealTime variants at O(1)
memory and would also make the latency claims of the paper reproducible in the
app. It is a larger change than this logging work and is not implemented here.

## Reading the logs

Every run writes `logs/inference/<timestamp>_<variant>_enhance.log` plus a JSON
sidecar with the same fields, and shows the same text in the **Detailed
Processing Log** box. The lines that matter for memory work:

```
predicted selective-scan workspace: 3.68 GB (peak of a single Mamba block pass, 256 x 16 state)
predicted total peak: 6.44 GB (scan workspace x 1.75 measured activation overhead)
gpu memory before inference: 10.79 GB free of 11.99 GB
[   5.947s] +   5.01 s | model_forward   | alloc 67 MB (+13 MB) peak 6.31 GB  single pass over 5.00 s, 251 frames
scan throughput: 2,897,544 scan iterations (46,360,704 state updates) in 24 kernel calls over 5.01 s -> 578,279 scan_iter/s
speed: 5.01 s of model time for 5.00 s of audio -> RTF 1.002 (1.00x realtime)
```

and the per-block table (`temporal(ms)`, `freq(ms)`, `block total(ms)`,
`scan iters`) which shows which layers actually ran (blocks above `exit_layer`
report 0) and where the time goes.

Useful ratios:

* `scan_iter/s` - throughput of the reference kernel. ~600 k/s at 16 kHz,
  ~580 k/s at 48 kHz on this GPU; it collapses to ~29 k/s once the driver
  starts spilling (the 15 GB / 12 s run).
* `RTF` - model seconds per audio second. `< 1` is real-time. 48 kHz mono 5 s
  measured 1.00; the same file at 16 kHz measured 0.31.
