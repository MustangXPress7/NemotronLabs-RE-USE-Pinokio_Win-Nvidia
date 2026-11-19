"""Phase 2 scaffold: native MLX / Core ML backend via ReuseKit (Swift).

The primary Mac backend is LiteRT/Metal (see mac_backend.py). This module
reserves the integration points for the native Apple path so the two can share
the same UI without either one blocking the other.

Source model pack (OpenMDW-1.1, commercial use permitted):
    https://huggingface.co/AILogDev/RE-USE-MLX-CoreML
Swift SDK, Mac CLI and samples:
    https://github.com/Corvelis/RE-USE-MLX-CoreML

Measured RTF (lower is faster, all offline):
    MacBook Air M2, 8-core GPU, MLX (`foldShared`) ........... 0.19824
    MacBook Air M2, 8-core GPU, Core ML + MLX (`foldSeparate`) 0.14692
    iPhone 17 Pro / A19 Pro, Core ML + MLX (`naxFullShared16`) 0.09480

Constraints that shape the integration:
    - 16 kHz mono, offline 5 s windows with 5% overlap (not a live microphone API).
    - All 30 stages and 16 FP32 recurrent states are retained.
    - Build environment used upstream: Xcode 27.2 / Swift 6.4.2 + Metal Toolchain.
    - The pack contains model_mlx.safetensors + hybrid_*.mlpackage + manifest.json
      + ModelCatalog.json + ReuseKit.zip.
"""

from __future__ import annotations

import os

HF_REPO = "AILogDev/REUSE-MLX-CoreML"
PACK_DIR = "ModelPack"
SDK_DIR = "ReuseKit"

MODES = {
    "mlx": "MLX GPU (todo el modelo en GPU)",
    "hybrid": "Core ML + MLX (CNN en Neural Engine, Mamba en MLX)",
}

SWIFT_API_EXAMPLE = """\
import ReuseKit

let audio = try ReuseAudioFile.read(inputURL, maximumSeconds: 300)
let engine = ReuseEngine(
    modelPack: ReuseModelPack(root: modelDirectory),
    cacheDirectory: cacheDirectory
)
let result = try await engine.enhance(audio.samples, mode: .mlx)
try ReuseAudioFile.write(result.samples, to: outputURL)
await engine.release()
"""

SWIFT_PROBE_COMMAND = """\
hf download AILogDev/REUSE-MLX-CoreML --local-dir ModelPack
# extraer ReuseKit.zip junto a ModelPack, luego:
cd ReuseKit
xcrun swift run --build-system swiftbuild -c release reuse-probe \\
  ../ModelPack input.wav output mlx 3
"""


def is_ready() -> bool:
    """True when both the model pack and the Swift SDK are present."""
    return os.path.isdir(PACK_DIR) and os.path.isdir(SDK_DIR)


def prepare_instructions() -> str:
    """Human-readable steps to make this backend available."""
    return SWIFT_PROBE_COMMAND


def enhance(*args, **kwargs):
    """Not wired yet.

    ReuseKit ships a Swift API, not a Python module. Exposing it inside the
    Gradio UI needs a small local bridge (a Swift CLI that reads/writes WAV and
    prints timing JSON, called with subprocess). Until that bridge exists the
    UI must keep LiteRT as the only selectable backend.
    """
    raise NotImplementedError(
        "The native MLX backend is prepared but not connected. "
        "See README_MAC.md ('Fase 2') and mlx_backend.prepare_instructions(). "
        "Use the LiteRT/Metal backend in the meantime."
    )
