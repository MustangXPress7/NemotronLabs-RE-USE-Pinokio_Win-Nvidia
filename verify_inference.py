"""End-to-end inference smoke test for the RE-USE launcher.

Generates a deterministic noisy WAV, runs it through the real model on the GPU,
and checks that the enhanced output is a readable, non-empty WAV file.

Prints a JSON summary and exits non-zero on failure. Safe to run repeatedly.
"""

import json
import math
import os
import sys
import tempfile
import time
import wave


def make_test_wav(path, sample_rate=16000, seconds=3.0):
    """Write a mono WAV with a sine tone plus deterministic noise."""
    import numpy as np

    rng = np.random.default_rng(0)
    frames = int(sample_rate * seconds)
    t = np.arange(frames, dtype=np.float64) / sample_rate
    tone = 0.25 * np.sin(2.0 * math.pi * 440.0 * t)
    noise = 0.05 * rng.standard_normal(frames)
    signal = np.clip(tone + noise, -1.0, 1.0)
    pcm = (signal * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    return sample_rate, frames


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")

    result = {
        "ok": False,
        "device": None,
        "input_sr": None,
        "output_path": None,
        "output_frames": 0,
        "elapsed_seconds": None,
        "errors": [],
    }

    temp_dir = tempfile.mkdtemp(prefix="reuse_verify_")
    input_path = os.path.join(temp_dir, "verify_input.wav")

    try:
        input_sr, _ = make_test_wav(input_path)
    except Exception as exc:
        result["errors"].append("could not write test WAV: {}".format(exc))
        return finish(result)

    try:
        import torch

        result["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

        import app

        started = time.time()
        with torch.no_grad():
            _original, enhanced_path, figure = app.enhance(input_path, "", "")
        result["elapsed_seconds"] = round(time.time() - started, 3)

        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception:
            pass

        if not enhanced_path or not os.path.exists(enhanced_path):
            result["errors"].append("enhance() did not return an existing output file")
            return finish(result)

        result["output_path"] = enhanced_path
        result["input_sr"] = input_sr

        with wave.open(enhanced_path, "rb") as handle:
            result["output_frames"] = handle.getnframes()
            result["output_sr"] = handle.getframerate()

        if result["output_frames"] <= 0:
            result["errors"].append("enhanced WAV has zero frames")
    except Exception as exc:
        result["errors"].append("inference failed: {}".format(exc))

    return finish(result)


def finish(result):
    result["ok"] = not result["errors"]
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        sys.stderr.write("Inference verification failed:\n")
        for error in result["errors"]:
            sys.stderr.write("  - {}\n".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
