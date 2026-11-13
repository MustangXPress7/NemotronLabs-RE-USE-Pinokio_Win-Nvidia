# RE-USE Speech Enhancement - Pinokio App

Pinokio launcher for the **NVIDIA RE-USE Speech Enhancement** model.

- Original Space: https://huggingface.co/spaces/nvidia/RE-USE
- Model: https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE

## Features

- Universal speech enhancement for noise, reverberation, clipping, and bandwidth limitation
- Accepts 8 kHz to 48 kHz audio; the STFT is scaled to the input sample rate
- Before/after spectrogram comparison
- Chunk-wise inference for the Standard family (overlap-add) and a frame-by-frame
  streaming path for the RealTime family that keeps memory flat over the file length
  (ported from `online_inference.py`); when the single-pass peak would exceed free
  VRAM, streaming switches on by itself
- Target and pre-low-pass sample-rate sliders with the same bandwidth-extension
  semantics as the original NVIDIA app (process and write at the target rate)
- Detailed per-run logs: pipeline stage timings, GPU memory, scan/RTF and a
  per-block model profile (see [OOM_ANALYSIS.md](OOM_ANALYSIS.md))
- NVIDIA-only: the launcher refuses to install without an NVIDIA GPU

## Quick Start with Pinokio

1. Install Pinokio Desktop from https://pinokio.computer
2. Add this folder as an app (File -> Add App -> Select this folder)
3. Click **Install** — creates a conda environment at `.env` (Python 3.10), installs
   FFmpeg, PyTorch cu128, Triton, and a pure-Python `mamba-ssm`, then verifies the GPU
4. Click **Run RE-USE** and open the captured local URL
5. Use **Verify GPU** at any time to re-check the CUDA stack

The model downloads from Hugging Face on first use.

## Using the app

1. Pick a **Model Variant**: `Standard` (30 blocks, chunked overlap-add) for
   quality and long files, `RealTime` / `Whisper` (12 blocks) for low latency.
2. Upload or record one file, or select several files for **Batch Enhance All Files**.
3. Press **Enhance**. The **Detailed Processing Log** shows per-stage timings,
   GPU memory and scan/RTF; **Open Output Folder** and **Open Logs Folder** open
   the results and the logs.
4. `Exit layer` and `Look ahead frames` only apply to the RealTime/Whisper
   variants; lower `Exit layer` is faster, look-ahead trades latency for quality.
5. **Streaming** (RealTime/Whisper only) runs the model frame by frame with cached
   causal-conv/Mamba states so memory stays flat however long the file is. It is
   slower per second of audio than a single pass, so it is opt-in, but it turns on
   by itself when the predicted single-pass peak would exceed free VRAM.

Notes:

- The **Target sample rate** slider makes the model run and the output play at
  that rate. The **Pre low-pass** slider is applied first (the original NVIDIA
  bandwidth-extension order), and only takes effect when a target rate is set;
  both are off at 0. Processing at a lower rate also shrinks the frequency axis,
  which cuts memory.
- Stereo files are processed per channel (chunked variants) or as a batch
  (RealTime), which doubles memory or runtime. The model expects mono.

## API

The app exposes named Gradio endpoints, so it can be driven without the browser
once the launcher is running (the port is the one Pinokio shows in the terminal):

| Endpoint | Inputs | Outputs |
|----------|--------|---------|
| `/unified_enhance` | `audio_path, lp_sr, target_sr, variant_id, chunk_size_s, overlap_portion, window_type, normalization, exit_layer, look_ahead_frames, streaming` | `input_audio, enhanced_audio, spectrograms, detailed_log` |
| `/batch_enhance_files` | `batch_files, ...same eleven...` | `input_audio, enhanced_audio, spectrograms, batch_log, detailed_log` |
| `/open_output_folder`, `/open_logs_folder` | none | none |

`variant_id` accepts either the registry id (`standard`, `laughter`, `bf16`,
`realtime`, `whisper`) or the dropdown label. `target_sr` makes the model run and
write at that rate; `lp_sr` resamples down to that rate first and is ignored when
`target_sr` is 0 (set both to 0 to disable). `streaming` (RealTime/Whisper only)
forces the frame-by-frame path and is also enabled automatically when the
predicted single-pass peak exceeds free VRAM. Discover the live schema at
`GET http://127.0.0.1:<port>/gradio_api/info`.

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/", verbose=False)
input_audio, enhanced_audio, spectrogram, log = client.predict(
    handle_file("noisy.wav"),  # audio_path
    0,                          # lp_sr (pre low-pass, needs target_sr)
    0,                          # target_sr (0 = off, keeps input rate)
    "realtime",                 # variant_id
    3,                          # chunk_size_s (standard only)
    0.5,                        # overlap_portion (standard only)
    "Hann",                     # window_type
    1.0,                        # normalization
    8,                          # exit_layer (realtime only)
    0,                          # look_ahead_frames (realtime only)
    False,                      # streaming (realtime only; auto if tight)
    api_name="/unified_enhance",
)
print(enhanced_audio)
print(log)
```

### JavaScript

```javascript
const base = "http://127.0.0.1:7860";

const { event_id } = await fetch(`${base}/gradio_api/call/unified_enhance`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    data: [
      { path: "/absolute/path/noisy.wav" }, // audio_path
      0, 0,                                 // lp_sr, target_sr
      "realtime", 3, 0.5, "Hann", 1.0, 8, 0,
      false,                                // streaming
    ],
  }),
}).then((r) => r.json());

const stream = await fetch(`${base}/gradio_api/call/unified_enhance/${event_id}`);
for (const line of (await stream.text()).split("\n")) {
  if (line.startsWith("data:")) {
    const payload = JSON.parse(line.slice(5));
    console.log(payload[1], payload[3]); // enhanced audio, detailed log
  }
}
```

### curl

```bash
BASE=http://127.0.0.1:7860

EVENT=$(curl -s -X POST "$BASE/gradio_api/call/unified_enhance" \
  -H "Content-Type: application/json" \
  -d '{"data":[{"path":"/absolute/path/noisy.wav"},0,0,"standard",3,0.5,"Hann",1.0,8,0,false]}' \
  | sed -E 's/.*"event_id":"([^"]+)".*/\1/')

curl -sN "$BASE/gradio_api/call/unified_enhance/$EVENT"
```

## Requirements

- Windows 10/11 or Linux, x64
- NVIDIA GPU with a driver that supports CUDA 12.8 (no separate CUDA Toolkit needed)
- Several GB of disk for PyTorch/CUDA wheels and dependencies
- Audio readable by `soundfile` (WAV, FLAC, OGG, MP3). `.m4a` and video containers
  are not supported — convert to WAV first

## Verification

Install ends with `python verify_gpu.py`, which prints a JSON report and exits
non-zero on failure. The **Verify GPU** menu entry re-runs it.

## Detailed processing logs

Every run prints a **Detailed Processing Log** in the UI and writes the same
text plus a JSON sidecar to `logs/inference/`. The **Open Logs Folder** button
opens that directory. One run looks like this:

```
[   0.095s] info   stft (scaled to input rate): n_fft=640 hop=320 win=640 -> 321 bins, 50.0 frames/s
[   0.095s] info   encoder grid: STFT T=251 -> scan T=251 x F=161 (time stride 1; d_inner=256, d_state=16)
[   0.653s] info   predicted selective-scan workspace: 1.23 GB (peak of a single Mamba block pass, 256 x 16 state)
[   0.653s] info   predicted total peak: 2.16 GB (scan workspace x 1.75 measured activation overhead)
[   0.868s] +  213.3 ms | stft             | alloc 53 MB (+1 MB) peak 54 MB   321 bins x 251 frames
[   2.411s] +    1.54 s | model_forward    | alloc 63 MB (+10 MB) peak 2.16 GB  single pass over 5.00 s, 251 scan frames
[   2.498s] info   scan throughput: 969,864 scan iterations (15,517,824 state updates) in 24 kernel calls
                   over 1.54 s of model time -> 628,903 scan_iter/s
[   2.498s] info   speed: 1.54 s of model time for 5.00 s of audio -> RTF 0.308 (3.24x realtime)
```

`scan T` is the **coded** time grid the Mamba stack sees. The RealTime encoder
keeps it equal to the STFT frame count (`T`), the Standard encoder divides it
by 4, so both families run the scan at 50 frames per second. The memory
prediction uses `scan T`, not the raw STFT frame count; see
[OOM_ANALYSIS.md](OOM_ANALYSIS.md#3-f_enc-and-t_scan-from-the-scaled-stft).

Followed by a per-block table with the temporal/frequency Mamba time of every
block and its scan iteration count. Blocks above the selected `exit_layer` show
zeros, which makes early-exit behaviour visible.

What the numbers mean:

| Field | Meaning |
|-------|---------|
| stage time | wall time of that step, measured after `torch.cuda.synchronize()` |
| `alloc` / `peak` | torch allocator usage before/after and the run peak |
| `scan T` | coded time frames the Mamba stack processes (RealTime: `T`, Standard: `T/4`) |
| `scan_iter/s` | selective-scan state iterations per second (the kernel's throughput) |
| `RTF` | model seconds per audio second; `< 1` is faster than real time |

The RealTime log lines are the fastest way to explain an OOM: see
[OOM_ANALYSIS.md](OOM_ANALYSIS.md) for the full analysis and the measured
memory table.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Check `predicted total peak` in the log. Tick **Streaming** for RealTime/Whisper to keep memory flat (it also auto-enables when the peak will not fit), split the file or convert to mono, resample to a lower target rate, or use the Standard variant. See [OOM_ANALYSIS.md](OOM_ANALYSIS.md) |
| RealTime takes minutes instead of seconds | On Windows the driver can spill to shared system memory instead of raising OOM; the log warns and auto-switches to streaming when the predicted peak exceeds free VRAM |
| Streaming is slower than single-pass | Expected: streaming runs one frame at a time with cached states (memory-flat but not real time in this pure-Python/PyTorch fallback setup). Use single-pass when it fits, or lower the target rate / exit layer |
| `torch.cuda.is_available() is False` | Install a recent NVIDIA driver with CUDA 12.8 support, then run **Verify GPU** |
| `LibsndfileError` on `.m4a` or video | Convert the file to WAV first |
| Mamba import error | Re-run **Install**; do not build `causal-conv1d` |

## License

The RE-USE model is licensed under NVIDIA's custom license. See the
[original space](https://huggingface.co/spaces/nvidia/RE-USE) for details.

## Credits

- **Model**: NVIDIA Nemotron Labs
- **Architecture**: SEMamba (Mamba-SSM for speech enhancement)
- **Original Space**: https://huggingface.co/spaces/nvidia/RE-USE
- **Pinokio Launcher**: this repository
