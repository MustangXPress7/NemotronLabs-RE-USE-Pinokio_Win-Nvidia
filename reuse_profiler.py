"""Inference instrumentation for the RE-USE launcher.

Two independent pieces:

* ``InferenceLogger`` -- a structured, timestamped run log. Every pipeline
  stage is timed, GPU memory is sampled before/after each stage, and the whole
  run is written to ``logs/inference/<timestamp>_<name>.log`` plus a JSON
  sidecar so failures can be audited after the fact.
* ``ModuleProfiler`` -- read-only forward hooks on the SEMamba stack that time
  the encoder, every ``TSMamba`` block (temporal and frequency halves) and the
  decoders, and count the number of selective-scan state updates. That count
  divided by the forward wall time gives ``scan_iter/s``, the throughput number
  that explains why the streaming model becomes memory bound in single-pass
  mode.

Both are strictly diagnostic: hooks never modify tensors, every measurement is
wrapped in ``try/except``, and a profiler failure can never break inference.

Output is ASCII only on purpose: the launcher's stdout is captured by Pinokio
on Windows, where a stray non-ASCII character would raise UnicodeEncodeError.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import json
import os
import re
import time

try:  # torch is present in the app, but keep this module importable standalone
    import torch
except Exception:  # pragma: no cover - defensive
    torch = None

_MB = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _ascii(text):
    """Return ``text`` restricted to ASCII so Windows consoles never choke."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _fmt_mb(mb):
    if mb is None:
        return "    n/a"
    if mb >= 1024.0:
        return f"{mb / 1024.0:5.2f} GB"
    return f"{mb:5.0f} MB"


def _fmt_duration(seconds):
    if seconds is None:
        return "     n/a"
    if seconds >= 1.0:
        return f"{seconds:7.2f} s"
    return f"{seconds * 1000.0:7.1f} ms"


def _fmt_rate(value, unit):
    if value is None:
        return "n/a"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M{unit}"
    if value >= 1e3:
        return f"{value / 1e3:.2f} k{unit}"
    return f"{value:.0f} {unit}"


def gpu_memory():
    """Snapshot of CUDA allocator + device memory, in MiB. Empty dict on CPU."""
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        dev = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(dev)
        return {
            "allocated_mb": torch.cuda.memory_allocated(dev) / _MB,
            "reserved_mb": torch.cuda.memory_reserved(dev) / _MB,
            "peak_mb": torch.cuda.max_memory_allocated(dev) / _MB,
            "free_mb": free / _MB,
            "total_mb": total / _MB,
        }
    except Exception:
        return {}


def _sync():
    """Block until the GPU queue is empty so wall times are real."""
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _reset_peak():
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# STFT / scan geometry (mirrors app.py and the model's encoder strides)
# ---------------------------------------------------------------------------
def _make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def scan_time_frames(t_frames, time_stride=1, time_kernel=1, time_pad=0):
    """Time frames the encoder hands to the Mamba stack.

    The scan workspace depends on the *coded* time grid, not on the STFT frame
    count. The generator (Standard) or the encoder (RealTime) adds
    ``time_pad`` frames of zero padding on the time axis, then a conv with
    ``time_kernel`` taps and ``time_stride`` downsamples it:

        t_scan = floor((t_frames + time_pad - time_kernel) / time_stride) + 1

    * Standard: +2 frames, (1, 3) conv with stride (4, 2) -> t_scan ~= T / 4.
    * RealTime: +2 frames, (3, 3) conv with stride (1, 2) -> t_scan == T.

    So both families reach the Mamba stack at 50 coded frames/s, and the scan
    cost per second of audio is the same. The RealTime OOM comes from feeding
    the whole file, not from a wider coded grid.
    """
    if t_frames is None:
        return None
    stride = max(1, int(time_stride or 1))
    kernel = max(1, int(time_kernel or 1))
    pad = max(0, int(time_pad or 0))
    return (int(t_frames) + pad - kernel) // stride + 1


def stft_geometry(cfg, sample_rate, num_samples=None, time_stride=1,
                  time_kernel=1, time_pad=0):
    """Resolve the STFT sizes and encoder grid for a given input sample rate.

    ``app.py`` scales ``n_fft``/``hop``/``win`` so the model keeps seeing the
    same time resolution it was trained on. The frequency axis, however, grows
    with the input sample rate, and that is the axis that drives the scan
    workspace. ``t_scan`` is the coded time grid the Mamba stack actually sees
    (see ``scan_time_frames``); ``t_frames`` is the raw STFT frame count.
    """
    stft = cfg["stft_cfg"]
    model_sr = stft["sampling_rate"]
    n_fft = _make_even(stft["n_fft"] * sample_rate // model_sr)
    hop = _make_even(stft["hop_size"] * sample_rate // model_sr)
    win = _make_even(stft["win_size"] * sample_rate // model_sr)
    freq_bins = n_fft // 2 + 1
    # Both generators append 2 zero frequency bins, then DenseEncoder applies a
    # (1, 3) conv with stride (1, 2) -> floor((freq_bins - 1) / 2) + 1.
    f_enc = (freq_bins - 1) // 2 + 1
    geom = {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop": hop,
        "win": win,
        "freq_bins": freq_bins,
        "f_enc": f_enc,
        "fps": sample_rate / hop if hop else 0.0,
        "time_stride": max(1, int(time_stride or 1)),
    }
    if num_samples is not None:
        # center=True adds n_fft // 2 of padding on both sides.
        geom["t_frames"] = int(num_samples) // hop + 1
        geom["t_scan"] = scan_time_frames(
            geom["t_frames"], time_stride, time_kernel, time_pad
        )
        geom["duration_s"] = float(num_samples) / float(sample_rate) if sample_rate else 0.0
    return geom


def estimate_scan_workspace_mb(t_scan, f_enc, d_inner, d_state, dtype_bytes=4, batch=1):
    """Peak transient allocation of one ``selective_scan_ref`` block pass.

    mamba's pure-Python reference kernel materialises two float32 tensors of
    shape ``[batch, d_inner, seq, d_state]`` (``deltaA`` and ``deltaB_u``).
    ``t_scan`` must be the coded time grid (see ``scan_time_frames``), not the
    raw STFT frame count. For both families:

      * temporal Mamba: batch = f_enc, seq = t_scan
      * frequency Mamba: batch = t_scan, seq = f_enc, called twice
        (forward + backward), but the calls do not overlap in time

    All three calls share the same ``batch * seq = t_scan * f_enc`` product,
    so the peak of a single block pass is ``2 tensors * 4 bytes * product *
    d_inner * d_state``, times the number of audio channels (the encoder keeps
    channels in the batch dimension). Blocks run sequentially, so that is also
    the run peak.
    """
    if not t_scan or not f_enc:
        return 0.0
    per_call_bytes = (
        2.0 * max(1, int(batch)) * t_scan * f_enc * d_inner * d_state * dtype_bytes
    )
    return per_call_bytes / _MB


def model_scan_shape(cfg):
    """``(d_inner, d_state)`` straight from the recipe."""
    mc = cfg["model_cfg"]
    hid = mc.get("hid_feature") or 0
    expand = mc.get("expand") or 1
    return int(hid * expand), int(mc.get("d_state") or 0)


# Measured on an RTX 4070 (12 GB) with realtime, mono, exit_layer 8, 5 s clips:
#   8 kHz  predicted 0.62 GB -> peak 1.12 GB (1.81x)
#   16 kHz predicted 1.23 GB -> peak 2.16 GB (1.76x)
#   48 kHz predicted 3.68 GB -> peak 6.31 GB (1.71x)
# The extra 1.7-1.8x is the encoder/decoder activation stack plus allocator
# granularity; it scales with f_enc, so it is stable across sample rates.
# Cross-check on the chunked Standard family (3 s, 16 kHz stereo, 3 s chunks):
#   t_scan 151, predicted 1.30 GB -> measured 1.32 GB (1.77x).
MEMORY_OVERHEAD_FACTOR = 1.75


def predict_peak_mb(t_scan, f_enc, d_inner, d_state, batch=1):
    """Calibrated peak estimate: scan workspace * measured overhead factor."""
    return estimate_scan_workspace_mb(
        t_scan, f_enc, d_inner, d_state, batch=batch
    ) * MEMORY_OVERHEAD_FACTOR


# ---------------------------------------------------------------------------
# Run logger
# ---------------------------------------------------------------------------
class InferenceLogger:
    """Timestamped stage log with GPU memory sampling and disk persistence."""

    def __init__(self, run_name="enhance", log_dir="logs/inference", enabled=True,
                 echo=True, console_prefix="[reuse]"):
        self.run_name = run_name
        self.log_dir = log_dir
        self.enabled = enabled
        self.echo = echo
        self.console_prefix = console_prefix
        self.lines = []
        self.meta = {}
        self.stages = []
        self.extras = {}
        self.run_id = None
        self.log_path = None
        self.json_path = None
        self._t0 = None

    # -- lifecycle ---------------------------------------------------------
    def begin(self, **meta):
        """Start a run: reset peak stats, stamp the clock, print the header."""
        self._t0 = time.perf_counter()
        self.lines = []
        self.stages = []
        self.extras = {}
        self.meta = {k: v for k, v in meta.items() if v is not None}
        stamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}_{self.run_name}"
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.run_id)
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = os.path.abspath(os.path.join(self.log_dir, safe + ".log"))
            self.json_path = os.path.abspath(os.path.join(self.log_dir, safe + ".json"))
        except Exception:
            self.log_path = None
            self.json_path = None
        _reset_peak()

        self._emit("=" * 78)
        self._emit(f" RE-USE inference run {self.run_id}")
        self._emit("=" * 78)
        for key, value in self.meta.items():
            self._emit(f" {key:<18}: {value}")
        mem = gpu_memory()
        if mem:
            self._emit(
                f" {'GPU':<18}: {_fmt_mb(mem['total_mb'])} total, "
                f"{_fmt_mb(mem['free_mb'])} free, "
                f"{_fmt_mb(mem['reserved_mb'])} reserved by torch"
            )
        self._emit("-" * 78)
        return self.run_id

    def finish(self, summary=None):
        """Write the log + JSON sidecar and return the text for the UI."""
        if not self.enabled:
            return ""
        elapsed = self.elapsed
        mem = gpu_memory()
        self._emit("-" * 78)
        peak = mem.get("peak_mb")
        self._emit(
            f" TOTAL {_fmt_duration(elapsed)} | peak torch allocation {_fmt_mb(peak)}"
        )
        if summary:
            for line in str(summary).splitlines():
                self._emit(f" {line}")
        self._emit(f" log: {self.log_path or '(not written)'}")
        self._write()
        return self.text()

    @property
    def elapsed(self):
        return (time.perf_counter() - self._t0) if self._t0 else 0.0

    def text(self):
        return "\n".join(self.lines)

    # -- emitting ----------------------------------------------------------
    def _emit(self, line, echo=True):
        line = _ascii(line)
        self.lines.append(line)
        if echo and self.echo:
            print(f"{self.console_prefix} {line}", flush=True)

    def info(self, message):
        if not self.enabled:
            return
        self._emit(f"[{self.elapsed:8.3f}s] info   {_ascii(message)}")

    def warn(self, message):
        if not self.enabled:
            return
        self._emit(f"[{self.elapsed:8.3f}s] WARN   {_ascii(message)}")

    def block(self, title, lines):
        """Append a titled block of pre-formatted lines."""
        if not self.enabled:
            return
        self._emit(f" {title}")
        for line in lines:
            self._emit(f"   {line}")

    def set(self, key, value):
        self.extras[key] = value

    @contextlib.contextmanager
    def stage(self, name, note=None, **extra):
        """Time a pipeline stage and record the GPU memory it needed."""
        if not self.enabled:
            yield None
            return
        _sync()
        t0 = time.perf_counter()
        m0 = gpu_memory()
        record = {
            "name": name,
            "note": note,
            "started_s": t0 - self._t0,
            "extra": dict(extra),
            "ok": True,
        }
        try:
            yield record
        except BaseException as exc:
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _sync()
            dt = time.perf_counter() - t0
            m1 = gpu_memory()
            record["seconds"] = dt
            if m0 and m1:
                record["alloc_mb"] = m1.get("allocated_mb")
                record["reserved_mb"] = m1.get("reserved_mb")
                record["peak_mb"] = m1.get("peak_mb")
                record["delta_mb"] = m1.get("allocated_mb", 0.0) - m0.get("allocated_mb", 0.0)
                mem_txt = (
                    f"alloc {_fmt_mb(m1.get('allocated_mb'))}"
                    f" ({record['delta_mb']:+.0f} MB)"
                    f" peak {_fmt_mb(m1.get('peak_mb'))}"
                )
            else:
                mem_txt = "cpu"
            note_txt = f"  {_ascii(note)}" if note else ""
            status = "" if record["ok"] else "  FAILED: " + record.get("error", "")
            self._emit(
                f"[{self.elapsed:8.3f}s] +{_fmt_duration(dt)} | "
                f"{name:<22} | {mem_txt}{note_txt}{status}"
            )
            self.stages.append(record)

    # -- persistence -------------------------------------------------------
    def _write(self):
        try:
            if self.log_path:
                with open(self.log_path, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(self.lines) + "\n")
        except Exception as exc:  # pragma: no cover - disk issues only
            print(f"{self.console_prefix} could not write log: {exc}", flush=True)
        try:
            if self.json_path:
                payload = {
                    "run_id": self.run_id,
                    "meta": self.meta,
                    "extras": self.extras,
                    "stages": self.stages,
                    "total_seconds": self.elapsed,
                }
                with open(self.json_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, default=str)
        except Exception as exc:  # pragma: no cover
            print(f"{self.console_prefix} could not write json: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Module profiler
# ---------------------------------------------------------------------------
_ENCODER_RE = re.compile(r"^dense_encoder$")
_BLOCK_RE = re.compile(r"^TSMamba\.(\d+)$")
_TIME_RE = re.compile(r"^TSMamba\.(\d+)\.time_mamba$")
_FREQ_RE = re.compile(r"^TSMamba\.(\d+)\.freq_mamba$")
_SCAN_RE = re.compile(
    r"^TSMamba\.(\d+)\.(time_mamba|freq_mamba)\.(forward_blocks|backward_blocks)$"
)
_DECODER_RE = re.compile(
    r"^(?:(?:mask|phase)_decoder$|(?:mask|phase)_decoder_list\.\d+$)"
)
_DECODER_LAYER_RE = re.compile(r"^(?:mask|phase)_decoder_list\.(\d+)$")


def _shape_of(tensor):
    try:
        return list(tensor.shape)
    except Exception:
        return None


class ModuleProfiler:
    """Forward-hook profiler for the SEMamba encoder / Mamba stack / decoders."""

    def __init__(self, model, enabled=True, echo=True, console_prefix="[reuse]"):
        self.model = model
        self.enabled = enabled and model is not None
        self.echo = echo
        self.console_prefix = console_prefix
        self.entries = {}
        self._handles = []
        self._use_cuda_events = bool(
            torch is not None and torch.cuda.is_available()
        )

    # -- registration ------------------------------------------------------
    def _register(self, key, module, group, block=None):
        entry = {
            "key": key,
            "group": group,
            "block": block,
            "module": type(module).__name__,
            "calls": 0,
            "starts": [],
            "ends": [],
            "iterations": 0,
            "state_updates": 0,
            "in_shape": None,
            "out_shape": None,
            "d_state": getattr(module, "d_state", None),
        }
        self.entries[key] = entry
        handle = module.register_forward_pre_hook(self._pre_hook(key))
        self._handles.append(handle)
        handle = module.register_forward_hook(self._post_hook(key))
        self._handles.append(handle)

    def start(self):
        if not self.enabled:
            return self
        try:
            self.stop()  # idempotent
            for name, module in self.model.named_modules():
                if _ENCODER_RE.match(name):
                    self._register(name, module, "encoder")
                elif _TIME_RE.match(name):
                    self._register(name, module, "time_mamba", int(_TIME_RE.match(name).group(1)))
                elif _FREQ_RE.match(name):
                    self._register(name, module, "freq_mamba", int(_FREQ_RE.match(name).group(1)))
                elif _SCAN_RE.match(name):
                    match = _SCAN_RE.match(name)
                    self._register(name, module, "scan", int(match.group(1)))
                elif _DECODER_RE.match(name):
                    layer = _DECODER_LAYER_RE.match(name)
                    self._register(
                        name, module, "decoder", int(layer.group(1)) if layer else None
                    )
                elif _BLOCK_RE.match(name):
                    self._register(name, module, "block", int(_BLOCK_RE.match(name).group(1)))
        except Exception as exc:  # pragma: no cover - never break inference
            self._log(f"profiler disabled: {exc}")
            self.stop()
            self.enabled = False
        return self

    def stop(self):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()
        return False

    # -- hooks -------------------------------------------------------------
    def _pre_hook(self, key):
        def hook(module, args):
            try:
                entry = self.entries[key]
                entry["calls"] += 1
                if self._use_cuda_events:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    entry["starts"].append(event)
                else:
                    entry["starts"].append(time.perf_counter())
                if entry["in_shape"] is None and args:
                    entry["in_shape"] = _shape_of(args[0])
                if entry["group"] == "scan" and args:
                    shape = _shape_of(args[0])
                    if shape and len(shape) >= 2:
                        pairs = int(shape[0]) * int(shape[1])
                        entry["iterations"] += pairs
                        if entry["d_state"]:
                            entry["state_updates"] += pairs * int(entry["d_state"])
            except Exception:
                pass
        return hook

    def _post_hook(self, key):
        def hook(module, args, output):
            try:
                entry = self.entries[key]
                if self._use_cuda_events:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    entry["ends"].append(event)
                else:
                    entry["ends"].append(time.perf_counter())
                if entry["out_shape"] is None:
                    entry["out_shape"] = _shape_of(output[0] if isinstance(output, tuple) else output)
            except Exception:
                pass
        return hook

    # -- reporting ---------------------------------------------------------
    def _elapsed_sum(self, entry):
        total = 0.0
        for start, end in zip(entry["starts"], entry["ends"]):
            try:
                if self._use_cuda_events:
                    total += start.elapsed_time(end) / 1000.0
                else:
                    total += end - start
            except Exception:
                pass
        return total

    def reset(self):
        """Clear collected samples but keep the hooks registered."""
        for entry in self.entries.values():
            entry.update(calls=0, starts=[], ends=[], iterations=0, state_updates=0,
                         in_shape=None, out_shape=None)

    def stats(self):
        total_scan_iters = sum(e["iterations"] for e in self.entries.values() if e["group"] == "scan")
        total_states = sum(e["state_updates"] for e in self.entries.values() if e["group"] == "scan")
        scan_calls = sum(e["calls"] for e in self.entries.values() if e["group"] == "scan")
        per_block = {}
        for entry in self.entries.values():
            block = entry["block"]
            if block is None:
                continue
            slot = per_block.setdefault(block, {"time_s": 0.0, "freq_s": 0.0, "block_s": 0.0,
                                                "time_iters": 0, "freq_iters": 0})
            seconds = self._elapsed_sum(entry)
            if entry["group"] == "time_mamba":
                slot["time_s"] += seconds
            elif entry["group"] == "freq_mamba":
                slot["freq_s"] += seconds
            elif entry["group"] == "block":
                slot["block_s"] += seconds
            elif entry["group"] == "scan":
                if ".time_mamba." in entry["key"]:
                    slot["time_iters"] += entry["iterations"]
                else:
                    slot["freq_iters"] += entry["iterations"]
        return {
            "scan_calls": scan_calls,
            "scan_iterations": total_scan_iters,
            "state_updates": total_states,
            "per_block": per_block,
            "encoder_s": sum(self._elapsed_sum(e) for e in self.entries.values()
                             if e["group"] == "encoder"),
            "decoder_s": sum(self._elapsed_sum(e) for e in self.entries.values()
                             if e["group"] == "decoder"),
            "decoder_calls": sum(e["calls"] for e in self.entries.values()
                                 if e["group"] == "decoder"),
            "encoder_shape_in": next((e["in_shape"] for e in self.entries.values()
                                      if e["group"] == "encoder"), None),
            "encoder_shape_out": next((e["out_shape"] for e in self.entries.values()
                                       if e["group"] == "encoder"), None),
        }

    def report_lines(self, forward_seconds=None):
        """Human-readable table; ``forward_seconds`` enables rate columns."""
        stats = self.stats()
        if not self.entries:
            return ["(no modules matched - profiler inactive)"]
        if not stats["scan_calls"]:
            return ["(model was never executed - profiler has no samples)"]

        rate_note = ""
        if forward_seconds:
            rate_note = (
                f" | forward {_fmt_duration(forward_seconds)}"
                f" -> {_fmt_rate(stats['scan_iterations'] / forward_seconds, 'scan_iter/s')},"
                f" {_fmt_rate(stats['state_updates'] / forward_seconds, 'state_updates/s')}"
            )

        lines = []
        shape_in = stats["encoder_shape_in"]
        shape_out = stats["encoder_shape_out"]
        if shape_in and shape_out:
            lines.append(
                f"encoder input {shape_in} -> coded {shape_out}"
            )
        lines.append(
            f"scan state updates: {stats['scan_iterations']:,} iterations x "
            f"d_state = {stats['state_updates']:,} in {stats['scan_calls']} kernel calls"
            + rate_note
        )
        lines.append("")
        lines.append(
            f"{'block':<7}{'temporal(ms)':>13}{'freq(ms)':>11}"
            f"{'block total(ms)':>18}{'scan iters':>13}"
        )
        for block in sorted(stats["per_block"]):
            slot = stats["per_block"][block]
            lines.append(
                f"{('#' + str(block)):<7}"
                f"{slot['time_s'] * 1000:13.1f}"
                f"{slot['freq_s'] * 1000:11.1f}"
                f"{slot['block_s'] * 1000:18.1f}"
                f"{slot['time_iters'] + slot['freq_iters']:13,}"
            )
        lines.append(
            f"{'total':<7}{'':>13}{'':>11}{'':>18}{stats['scan_iterations']:13,}"
        )
        lines.append("")
        lines.append(
            f"encoder {_fmt_duration(stats['encoder_s'])} | "
            f"decoders {_fmt_duration(stats['decoder_s'])} "
            f"({stats['decoder_calls']} module{'s' if stats['decoder_calls'] != 1 else ''} executed)"
        )
        return lines

    def log_to(self, logger, forward_seconds=None, title="MODULE PROFILE (forward hooks)"):
        if not self.enabled or logger is None or not logger.enabled:
            return
        try:
            logger.block(title, self.report_lines(forward_seconds=forward_seconds))
        except Exception as exc:
            logger.warn(f"module profile failed: {exc}")

    def _log(self, message):
        if self.echo:
            print(f"{self.console_prefix} {_ascii(message)}", flush=True)
