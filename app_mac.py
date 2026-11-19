"""Gradio UI for the Apple Silicon build of RE-USE.

Primary backend: LiteRT with the Metal GPU delegate (offline, 16 kHz mono).
The native MLX path is prepared in mlx_backend.py and is not selectable yet.

This file is intentionally separate from app.py so that shared fixes merged
from the Windows/Nvidia branch do not conflict with the Mac backend.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mac_backend import (
    GRAPH_LABELS,
    MODEL_SPECS,
    ReuseLiteRT,
    SR,
    list_available_models,
    read_audio,
    write_wav,
)

ABOUT = """\
# RE-USE · Apple Silicon (LiteRT / Metal)

Mejora de voz offline con los grafos LiteRT de NVIDIA convertidos por la comunidad.
**Solo archivos**: no es un filtro de micrófono en tiempo real.

Sube un clip, elige variante y grafo, y pulsa **Enhance**.
"""

NOTES = """\
- **Fastload** carga en ~10 s; **Main** en ~131 s pero es la opción de referencia.
- Todo se procesa a **16 kHz mono**; los archivos a otras tasas se remuestrean.
- Los grafos trabajan en chunks de 2 s con solape de 0,25 s (1,75 s de salto).
- En M4 Max, LiteRT/Metal tarda ~0,45 s por segundo de audio (más rápido que en tiempo real).
- Si el delegate Metal no está disponible, el backend cae a CPU (XNNPACK) automáticamente.
"""

OUTPUT_DIR = "output"


def _status_text() -> str:
    if platform.system() != "Darwin":
        return "⚠️ Este launcher está declarado solo para macOS/arm64. El backend puede no cargar aquí."
    available = list_available_models()
    lines = []
    for key, spec in MODEL_SPECS.items():
        graphs = available[key][spec[3]]
        for graph, ok in graphs.items():
            lines.append(f"- {spec[3]} / {graph}: {'✓' if ok else '✗ falta (re-run Install)'}")
    return "Modelos:\n" + "\n".join(lines)


def _spectrogram(samples, path: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.specgram(samples, NFFT=1024, Fs=SR, noverlap=768, cmap="magma")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_title("Espectrograma del resultado (16 kHz)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def enhance(audio_path, variant: str, graph: str, accel: str):
    if not audio_path:
        return None, "Primero sube un archivo de audio.", None

    try:
        x, original_sr = read_audio(audio_path)
    except Exception as exc:
        return None, f"No se pudo leer el audio: {exc}", None

    backend = None
    try:
        t_load = time.time()
        backend = ReuseLiteRT.load(variant, graph, accel)
        load_s = time.time() - t_load

        result = backend.enhance(x)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        wav_path = os.path.join(OUTPUT_DIR, f"{stamp}_{variant}_{graph}_enhanced.wav")
        write_wav(wav_path, result.samples)
        spec_path = _spectrogram(result.samples, os.path.join(OUTPUT_DIR, f"{stamp}_spec.png"))

        rtf = result.seconds / max(len(x) / SR, 1e-9)
        log = (
            f"Variante: {MODEL_SPECS[variant][3]}\n"
            f"Grafo: {graph} ({GRAPH_LABELS[graph]})\n"
            f"Backend: LiteRT / {result.accel}\n"
            f"Entrada: {original_sr} Hz -> {SR} Hz mono, {len(x) / SR:.2f} s\n"
            f"Chunk: {result.chunk_samples} muestras ({result.chunk_samples / SR:.2f} s), "
            f"{result.frames} frames, salto {backend.hop} muestras\n"
            f"Chunks procesados: {result.chunks}\n"
            f"Carga del modelo: {load_s:.2f} s\n"
            f"Inferencia: {result.seconds:.2f} s (RTF {rtf:.3f})\n"
            f"Salida: {wav_path}"
        )
        return wav_path, log, spec_path
    except Exception as exc:
        return None, f"Error durante la mejora: {type(exc).__name__}: {exc}", None
    finally:
        if backend is not None:
            backend.close()


def open_output_folder():
    folder = os.path.abspath(OUTPUT_DIR)
    os.makedirs(folder, exist_ok=True)
    if platform.system() == "Darwin":
        subprocess.run(["open", folder], check=False)
    return folder


with gr.Blocks(title="RE-USE Speech Enhancement · Apple Silicon") as demo:
    gr.Markdown(ABOUT)

    with gr.Accordion("Notas y consejos", open=False):
        gr.Markdown(NOTES)
        gr.Textbox(value=_status_text(), label="Estado de modelos", lines=6, interactive=False)

    with gr.Row():
        variant = gr.Dropdown(
            label="Variante",
            choices=[(spec[3], key) for key, spec in MODEL_SPECS.items()],
            value="standard",
            scale=2,
        )
        graph = gr.Dropdown(
            label="Grafo",
            choices=[(label, key) for key, label in GRAPH_LABELS.items()],
            value="fastload",
            scale=2,
        )
        accel = gr.Dropdown(
            label="Acelerador",
            choices=[("GPU / Metal", "gpu"), ("CPU / XNNPACK", "cpu")],
            value="gpu",
            scale=1,
        )
        enhance_btn = gr.Button("Enhance", variant="primary", scale=1)

    input_audio = gr.Audio(label="Audio de entrada (WAV, FLAC, OGG, MP3)", type="filepath")

    with gr.Row():
        output_audio = gr.Audio(label="Resultado", type="filepath")
        spectrogram = gr.Image(label="Espectrograma", type="filepath")

    detail_log = gr.Textbox(label="Log", lines=10, interactive=False)
    gr.Button("Abrir carpeta de salida").click(fn=open_output_folder, inputs=None, outputs=None)

    enhance_btn.click(
        fn=enhance,
        inputs=[input_audio, variant, graph, accel],
        outputs=[output_audio, detail_log, spectrogram],
    )


if __name__ == "__main__":
    demo.queue().launch(share=False)
