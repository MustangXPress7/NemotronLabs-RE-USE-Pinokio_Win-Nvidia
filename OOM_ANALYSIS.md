# Why the RealTime variants run out of memory

This note explains where the memory goes when the **RealTime** variants are used
in this app, why the **Standard** variants do not hit the same wall, and what the
upstream papers and model cards say about it. Every number below was measured on
this machine (RTX 4070, 12 GB) with the instrumentation added in
`reuse_profiler.py`; the raw runs are in `logs/inference/`.

## TL;DR

1. The RealTime family can run **one forward pass over the whole file** (the
   offline path) or **frame by frame** with cached states (the streaming path
   ported in `reuse_streaming.py`). The model was trained and published as a
   streaming model; the offline pass keeps a workspace proportional to
   `duration x frequency bins` instead of `O(1)`, so the app now auto-switches
   to streaming when the predicted single-pass peak would exceed free VRAM (and
   exposes a Streaming checkbox to force it). The Standard family stays chunked.
2. `mamba_ssm` is installed **without** the compiled `selective_scan_cuda`
   extension (see `reuse_runtime.py`), so every scan goes through
   `selective_scan_ref`, a pure-Python kernel that materialises two float32
   tensors of shape `[batch, d_inner, seq, d_state]` per call.
3. That product is what explodes:

   ```
   peak scan workspace ~= 2 tensors * 4 B * t_scan * f_enc * d_inner * d_state
   total peak          ~= 1.75 * peak scan workspace
   ```

   `t_scan` is the **coded** time grid the Mamba stack sees (see §3), and it
   grows with duration. `f_enc` grows with the **input sample rate** (48 kHz ->
   481 bins, 16 kHz -> 161, 8 kHz -> 81), and `d_inner = 256`, `d_state = 16`
   are fixed by the recipe.
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

### 1b. NVIDIA's own long-audio path is `inference_chunk.sh`

The Standard model card ships two batch scripts: `inference.sh` (single-pass) and
`inference_chunk.sh`. The card says to use the chunked one when the audio is
longer than ~30 s **or when the GPU runs out of memory**. `inference_chunk.sh`
is a thin wrapper over `inference_chunk.py`, whose loop is:

```python
chunk_size = int(5 * noisy_sr)              # 5 s
hop_length = int(0.5 * chunk_size)          # 50% overlap
window = torch.clamp(torch.hann_window(chunk_size), min=1e-7)
...
audio_enhanced[...] += audio_g * window
window_sum[...]    += window
...
audio_enhanced = audio_enhanced / window_sum
```

That is the same algorithm this app uses for the Standard family (the app
defaults to 3 s instead of 5 s, saves WAV instead of FLAC, clamps `window_sum`
to avoid division by zero, and adds `torch.cuda.empty_cache()` per chunk). In
other words, the chunking that protects Standard is not an invention of this
launcher: it is the official mitigation, ported almost line by line.

There is **no chunked script for the RealTime model**. The Real-time Space's
`app.py` is single-pass, exactly like this app's RealTime branch, so the OOM is
inherited from the official demo code. The official fix for RealTime is the
streaming path (`mamba.step()` with cached states), not chunking.

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

* temporal Mamba: `batch = f_enc`, `seq = t_scan`
* frequency Mamba: `batch = t_scan`, `seq = f_enc` (forward + backward, two
  calls, sequential)

All of them share the same `batch * seq = t_scan * f_enc`, which is why a
single formula covers the whole block. Blocks run one after another, so the
peak is one block pass, not the sum over blocks.

### 3. `f_enc` and `t_scan` from the scaled STFT

`app.py` scales the STFT to the input sample rate so the model keeps its time
resolution:

```
n_fft  = make_even(320 * sr / 8000)
hop    = make_even(160 * sr / 8000)
bins   = n_fft / 2 + 1
f_enc  = (bins - 1) / 2 + 1        # DenseEncoder (1,3) conv with stride (1,2)
T      = samples / hop + 1         # center=True
```

The scan does **not** read `T` directly. Both generators add 2 zero frames on
the time axis before the encoder, and the encoder downsamples time with a conv
(`reuse_profiler.scan_time_frames`):

```
t_scan = (T + time_pad - time_kernel) // time_stride + 1
```

| family | time pad | conv | stride | t_scan |
|--------|----------|------|--------|--------|
| Standard (`hop_40`) | +2 | (1, 3) | 4 | `(T + 1) // 4 + 1` ~= T / 4 |
| RealTime (`hop_160`) | +2 | (3, 3) | 1 | `T` |

Consequences:

* Both families reach the Mamba stack at **50 coded frames per second** (the
  Standard STFT runs at 200 fps and is divided by 4; RealTime runs at 50 fps
  and is not divided). So the scan cost *per second of audio* is the same for
  both models: the RealTime OOM is not caused by a wider coded grid.
* The Standard STFT frame count is 4x larger, but the app feeds it 3 s chunks,
  so its `t_scan` stays at ~151 and the peak is bounded.
* The RealTime branch feeds the whole file, so `t_scan = 50 * duration`.

Frequency and time scaling by input sample rate:

| input | hop | STFT frames/s | bins | f_enc | t_scan x f_enc per second |
|-------|-----|---------------|------|-------|---------------------------|
| 8 kHz | 160 | 50 | 161 | 81 | 50 x 81 |
| 16 kHz | 320 | 50 | 321 | 161 | 50 x 161 |
| 48 kHz | 960 | 50 | 961 | 481 | 50 x 481 |

The coded frame rate is constant at 50 fps for both families (Standard divides
its 200 fps STFT by 4), but the frequency axis scales linearly with the sample
rate. **Uploading 48 kHz audio costs 5.9x more scan memory than 8 kHz for the
same duration.**

### 4. Measured peak

Realtime, mono, `exit_layer=8`, `look_ahead=0`, 5 s clips, RTX 4070:

| input | t_scan x F | predicted scan | predicted peak | measured peak | model time |
|-------|------------|----------------|----------------|---------------|------------|
| 8 kHz | 251 x 81 | 0.62 GB | 1.09 GB | **1.12 GB** | 0.79 s |
| 16 kHz | 251 x 161 | 1.23 GB | 2.16 GB | **2.16 GB** | 1.54 s |
| 48 kHz | 251 x 481 | 3.68 GB | 6.44 GB | **6.31 GB** | 5.01 s |

Cross-check on the chunked side (Standard, 3 s chunk, 3 s / 16 kHz / stereo
input, one chunk per channel): STFT `T=601` -> `t_scan=151`, predicted peak
**1.30 GB**, measured peak **1.32 GB** (1.77x). That confirms the coded grid is
the one that sets memory, and that a 3 s chunk costs about the same as a 1 s
RealTime frame at the same sample rate, no matter how long the file is.

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
| **Streaming** (RealTime/Whisper, implemented) | Frame-by-frame `mamba.step()` with cached causal-conv/Mamba states: memory flat at ~140-160 MB, independent of duration | Slower model pass in this pure-Python/PyTorch fallback setup (see below) |
| Use **Standard** (chunked) for long files | Peak set by chunk size (measured 1.32 GB for a 3 s / 16 kHz stereo chunk, 1.30 GB predicted) | Not the low-latency model |
| Process at a lower **target rate** (slider) | Shrinks `f_enc` linearly: 16 kHz / 8 kHz cost ~3x / ~6x less scan memory than 48 kHz | Bandwidth above the target lost |
| Split the file into 5-8 s clips | Linear reduction for the single pass | Manual work; chunk joins |
| Mono | ~2x less memory | Channel loss |
| Lower `exit_layer` | Faster; peak unchanged for the single pass, and fewer cached blocks in streaming | Quality |
| `torch.cuda.empty_cache()` between chunks | Avoids fragmentation | Small overhead |

## Streaming mode (implemented)

`reuse_streaming.py` ports the official `online_inference.py` path from
`nvidia/Real-time_RE-USE`: causal convolutions keep a time cache, the temporal
Mamba runs through `mamba.step()` with cached conv/SSM states, and the frequency
Mamba (non-causal, full width) is evaluated per frame. Weights are shared by
reference with the loaded model.

Because this environment has no compiled `selective_scan_cuda` and the reference
kernel's Python loop is extremely expensive per call, `reuse_runtime.py` also
installs a chunked, closed-form selective scan (self-tested against the
reference) so the per-frame frequency path is practical. The log reports
per-frame latency, RTF and the streaming peak, and notes that the scan
throughput counter only covers the frequency path (the temporal Mamba goes
through `step`, outside the forward hooks).

Measured on a 5 s / 16 kHz / mono clip, RTX 4070, `exit_layer=8`,
`look_ahead=0`:

| mode | model time | RTF | peak GPU memory | max audio diff vs offline |
|------|------------|-----|-----------------|---------------------------|
| offline (single pass) | 1.49 s | 0.30 | 2211 MB | - |
| streaming (frame by frame) | 15.00 s | 3.00 | 156 MB | 1.12e-04 (mean 5.88e-06) |

Streaming trades throughput for a memory footprint that does not grow with the
file. The app exposes a **Streaming** checkbox for the RealTime/Whisper variants
and also enables streaming automatically when the predicted single-pass peak
exceeds free VRAM, to avoid an OOM or a WDDM spill into shared memory. Standard
keeps its chunked overlap-add path.

## Reading the logs

Every run writes `logs/inference/<timestamp>_<variant>_enhance.log` plus a JSON
sidecar with the same fields, and shows the same text in the **Detailed
Processing Log** box. The lines that matter for memory work:

```
stft (scaled to input rate): n_fft=1920 hop=960 win=1920 -> 961 bins, 50.0 frames/s
encoder grid: STFT T=251 -> scan T=251 x F=481 (time stride 1; d_inner=256, d_state=16)
predicted selective-scan workspace: 3.68 GB (peak of a single Mamba block pass, 256 x 16 state)
predicted total peak: 6.44 GB (scan workspace x 1.75 measured activation overhead)
gpu memory before inference: 10.79 GB free of 11.99 GB
[   5.947s] +   5.01 s | model_forward   | alloc 67 MB (+13 MB) peak 6.31 GB  single pass over 5.00 s, 251 scan frames
scan throughput: 2,897,544 scan iterations (46,360,704 state updates) in 24 kernel calls over 5.01 s -> 578,279 scan_iter/s
speed: 5.01 s of model time for 5.00 s of audio -> RTF 1.002 (1.00x realtime)
```

For the Standard family the same block reports the chunk budget instead:

```
encoder grid: STFT T=601 -> scan T=151 x F=161 (time stride 4; d_inner=256, d_state=16)
per-chunk scan workspace: 0.74 GB (estimated peak 1.30 GB) for scan T=151 x F=161 (STFT T=601); chunk size is the memory knob in this mode
[   4.984s] +   4.15 s | ch1 chunk 1/1 | alloc 54 MB (+16 MB) peak 1.31 GB
```

`scan T` (not `STFT T`) is the number to compare against the predicted memory,
and it is the number that must stay bounded.

and the per-block table (`temporal(ms)`, `freq(ms)`, `block total(ms)`,
`scan iters`) which shows which layers actually ran (blocks above `exit_layer`
report 0) and where the time goes.

Useful ratios:

* `scan_iter/s` - throughput of the reference kernel. ~600 k/s at 16 kHz,
  ~580 k/s at 48 kHz on this GPU; it collapses to ~29 k/s once the driver
  starts spilling (the 15 GB / 12 s run).
* `RTF` - model seconds per audio second. `< 1` is real-time. 48 kHz mono 5 s
  measured 1.00; the same file at 16 kHz measured 0.31.
