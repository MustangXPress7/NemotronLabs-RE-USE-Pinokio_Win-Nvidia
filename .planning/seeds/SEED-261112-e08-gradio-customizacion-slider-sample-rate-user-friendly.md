---
id: SEED-261112-e08
status: dormant
planted: 2026-11-12T00:00:00Z
planted_during: Phase 1: NVIDIA install and GPU truth
trigger_when: when relevant
scope: unknown
---

# SEED-261112-e08: ver si el gradio podría soportar alguna customización para, por ejemplo regular el sample rate manualmente con un slider y hacerlo más intuitivo por ejemplo... O cualquier otra opción que deje el modelo hacerle poderle liberar o plantear desde una modificación más "user-friendly".

## Why This Matters

_To be filled in. Run `/gsd-capture --seed --enrich SEED-261112-e08` to add context._

## When to Surface

**Trigger:** when relevant

This seed will surface during `/gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `/gsd-capture --seed --enrich SEED-261112-e08` to estimate effort.

## Breadcrumbs

- `app.py:193-194` — `target_sampling_rate` y `low_pass_sampling_rate` son `gr.Textbox` (candidatas a slider)
- `app.py:65` — `sampling_rate` leído de la config del modelo
- `app.py:14` — `import gradio as gr`
- `app.py:207` — `inputs=[input_audio, low_pass_sampling_rate, target_sampling_rate]`
- `pinokio.json:10` — tag "gradio"
- `.planning/phases/01-nvidia-install-and-gpu-truth/01-03-SUMMARY.md` y `01-05-SUMMARY.md` — gradio en tags
- `https://huggingface.co/spaces/nvidia/Real-time_RE-USE` — Hugging Face Space de Real-time RE-USE (modelos apuntados y parámetros de referencia)

## Notes

_Captured via one-shot seed capture. Enrich with trigger, why, and scope at your convenience._

Idea captured: hacer la interfaz Gradio más user-friendly permitiendo customizacións como regular el sample rate manualmente con un slider, y cualquier otra opción que haga el modelo más accesible / "liberable" desde la UI.

Referencias de código candidate: `app.py` (bloque `gr.Blocks`, `enhance_btn.click`, inputs `target_sampling_rate` / `low_pass_sampling_rate` como Textbox), `pinokio.js` (menú), `pinokio.json` (metadata).

Usar uno de los modelos apuntados por `https://huggingface.co/spaces/nvidia/Real-time_RE-USE` como base para las customizaciones, y añadir más parámetros expuestos desde la UI Gradio (más allá del sample rate: chunk size, overlap, ventana, normalización, etc.) para que el usuario pueda ajustar el comportamiento del modelo de forma más intuitiva.
