---
id: SEED-261112-mos
status: dormant
planted: 2026-11-12T00:00:00Z
planted_during: Phase 1: NVIDIA install and GPU truth
trigger_when: when relevant
scope: unknown
---

# SEED-261112-mos: viabilidad de usar Vulkan, ROCm, Metal o OpenVINO para ejecutar este modelo

## Why This Matters

_To be filled in. Run `/gsd-capture --seed --enrich SEED-261112-mos` to add context._

## When to Surface

**Trigger:** when relevant

This seed will surface during `/gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Unknown** — run `/gsd-capture --seed --enrich SEED-261112-mos` to estimate effort.

## Breadcrumbs

- `install.js:9` — `when: "{{gpu !== 'nvidia' || (platform !== 'win32' && platform !== 'linux')}}"` fail-fast sin GPU NVIDIA
- `install.js:56-59` — torch.js con `triton: true`, xformers/sageattention/flashattention false
- `app.py:39-43` — `torch.cuda.is_available()` raise SystemExit si no CUDA
- `pinokio.json:7-8` — `"arch": ["x64"]`, `"gpu": ["nvidia"]`
- `pinokio.json:6` — `"platform": ["win32", "linux"]`
- `reuse_runtime.py` — parche de mamba_selective_scan (CUDA-only path)
- `pinokio.json:6-8` — ninguna referencia a OpenVINO, Intel, Vulkan, ROCm o Metal; solo NVIDIA/CUDA
- `https://huggingface.co/spaces/nvidia/Real-time_RE-USE` — Hugging Face Space de Real-time RE-USE (modelos apuntados y parámetros de referencia)

## Notes

_Captured via one-shot seed capture. Enrich with trigger, why, and scope at your convenience._

Idea captured: evaluar si es viable portar/adaptar el modelo RE-USE para ejecutarse sobre backends no-CUDA: Vulkan (multiplataforma GPU), ROCm (AMD, Linux), Metal (Apple Silicon) o OpenVINO (Intel CPU/iGPU/dGPU), rompiendo el acoplamiento actual NVIDIA-only.

Usar uno de los modelos apuntados por `https://huggingface.co/spaces/nvidia/Real-time_RE-USE` como base para los tests de viabilidad en cada backend, y añadir más parámetros de evaluación (hardware objetivo, sistema operativo, versión de CUDA/Triton, tamaño de chunk, latencia, memoria GPU, throughput) para comparar el comportamiento entre NVIDIA CUDA y las alternativas Vulkan/ROCm/Metal/OpenVINO.

## Investigación de referencia (IA)

La viabilidad de ejecutar NVIDIA NemotronLabs RE-USE en Vulkan, Metal o OpenVINO es técnicamente posible, pero requiere trabajo manual de adaptación, ya que el modelo no ofrece soporte nativo inmediato fuera del ecosistema NVIDIA CUDA.

El principal cuello de botella es su arquitectura: utiliza un bloque Mamba bidireccional de 30 capas junto con transformadas de Fourier (STFT) y un codificador/decodificador convolucional. La implementación oficial depende de la librería mamba-ssm, cuyos núcleos (kernels) de escaneo selectivo (selective scan) están escritos específicamente en CUDA y Triton.

### Comparativa de viabilidad por plataforma

| Entorno | Viabilidad | Estado y esfuerzo necesario |
|---------|-----------|------------------------------|
| OpenVINO (Intel CPU/GPU/NPU) | Media - Alta | Requiere exportar el modelo a ONNX sustituyendo los kernels CUDA de Mamba por operaciones estándar de PyTorch (fallback puro). Intel ya soporta modelos basados en Mamba (como Falcon Mamba) en OpenVINO, pero la variante bidireccional y el preprocesado STFT de este modelo obligan a crear el grafo de exportación a mano. |
| Metal (Apple Silicon) | Media | No funciona de serie con PyTorch MPS debido a las extensiones CUDA de mamba-ssm. Sin embargo, es viable reescribiendo el bloque Mamba en MLX (el framework de Apple) o portando las operaciones a llama.cpp / ggml, que ya cuentan con soporte para bloques Mamba y aceleración Metal. |
| Vulkan (GPUs genéricas / móviles) | Baja - Media | No existe un exportador directo de PyTorch a Vulkan para esta arquitectura. La única vía práctica es integrar el modelo dentro de un motor basado en ggml (con backend Vulkan activo) o compilar el grafo mediante IREE / TVM, lo cual exige escribir shaders o mapeos personalizados para el escaneo selectivo bidireccional. |

### Principales retos técnicos para portarlo

- **Dependencia de mamba-ssm.** La implementación de referencia en Hugging Face importa directamente módulos compilados en CUDA. Para sacarlo de NVIDIA, primero debes reemplazar la función selective_scan_fn por una implementación equivalente en PyTorch puro (más lenta al exportar, pero trazable hacia ONNX o MLX).
- **Procesamiento bidireccional.** A diferencia de los modelos de lenguaje Mamba (que son causales y solo miran hacia atrás), RE-USE analiza el audio en ambos sentidos (pasado y futuro). Esto significa que el motor de destino debe ejecutar el escaneo tanto hacia adelante como hacia atrás, lo que impide usar en tiempo real streaming sin un búfer de ventana (chunking).
- **Transformadas de tiempo-frecuencia.** El archivo models/stfts.py del repositorio utiliza operaciones de espectrogramas (STFT/iSTFT) con números complejos que a veces fallan al convertirse a ONNX o OpenVINO, requiriendo reescribirlas mediante convoluciones 1D reales.

### Recomendación práctica

- **Si buscas rendimiento en CPU o hardware Intel.** La ruta más limpia es reescribir el forward de Mamba en PyTorch puro, exportarlo a ONNX (usando opset 17 o superior) y compilarlo con OpenVINO. Al pesar apenas 9,6 millones de parámetros, incluso sin optimizaciones extremas de kernels, correrá muy rápido en una CPU moderna.
- **Si desarrollas para macOS o iOS.** Portar los pesos (model.safetensors, ~39 MB) a MLX es la opción más directa para aprovechar los núcleos GPU de Apple mediante Metal sin pelear con conversiones intermedias.

Pendiente: definir el hardware o sistema operativo específico de despliegue (Mac con chip M, PC con Intel/AMD, dispositivos móviles) para detallar la ruta de exportación más directa.
