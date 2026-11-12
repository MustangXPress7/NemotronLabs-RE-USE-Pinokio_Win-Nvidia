# Modelos MLX / LiteRT descartados — Reserva para versión Mac (futura fase)

Este documento recopila los modelos RE-USE convertidos a MLX / LiteRT que **no son compatibles** con el launcher actual (NVIDIA CUDA + PyTorch + mamba-ssm), pero que son candidatos para una versión futura del launcher ejecutándose en Apple Silicon (Mac con chip M) mediante MLX o LiteRT.

## Modelos descartados (no compatibles con launcher NVIDIA CUDA)

| # | Modelo | Backend | Hardware objetivo | Formato | Tamaño | URL |
|---|--------|---------|-------------------|---------|--------|-----|
| 1 | `litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT` | LiteRT (TFLite) | Android / Mac Metal | `.tflite` | ~54 MB (main) / ~44 MB (fastload) | https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT |
| 2 | `faraday/re-use-mlx` | MLX | Apple Silicon | `.safetensors` (MLX) | ~38.6 MB | https://huggingface.co/faraday/re-use-mlx |
| 3 | `litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT` | LiteRT (TFLite) | Android / Mac Metal | `.tflite` | ~54 MB (main) / ~44 MB (fastload) | https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT |
| 4 | `appautomaton/re-use-semamba-mlx` | MLX | Apple Silicon | `.safetensors` (MLX) | ~38 MB | https://huggingface.co/appautomaton/re-use-semamba-mlx |
| 5 | `litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT` | LiteRT (TFLite) | Android / Mac Metal | `.tflite` | ~54 MB (main) / ~44 MB (fastload) | https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT |

*(Entrada 5 es duplicada de la entrada 3 — mismo modelo, mismo archivo.)*

## Por qué no son compatibles con el launcher actual

- El launcher actual usa **PyTorch + mamba-ssm + CUDA** en NVIDIA GPUs.
- Los modelos MLX requieren el runtime de **Apple MLX** (solo Apple Silicon).
- Los modelos LiteRT/TFLite requieren el runtime de **LiteRT / TFLite** (Android o Mac Metal).
- Ninguno de estos backends está disponible en el entorno actual del launcher (`install.js` solo instala PyTorch cu128, mamba-ssm, triton).

## Qué se necesitaría para una versión Mac

| Componente | Cambio requerido |
|------------|-----------------|
| Runtime Python | `mlx` o `ai-edge-litert` en lugar de `torch` + `mamba-ssm` |
| Modelo de inferencia | Reemplazar `SEMamba`/`SEMamba_decoder_list` por carga de `.mlxmodel` o `.tflite` |
| STFT/iSTFT | Reimplementar con operaciones compatibles con MLX/LiteRT (o mantener en NumPy/host como hace la conversión LiteRT) |
| Selective scan | La conversión MLX ya incluye una implementación de referencia (`selective_scan_ref` equivalente) |
| `device = "cuda"` | Cambiar a `device = "mps"` (Metal) o dejar en CPU para LiteRT |
| `pinokio.json` | Cambiar `"gpu": ["nvidia"]` a `"gpu": ["apple"]` o `"gpu": ["metal"]`; añadir `"platform": ["darwin"]` |
| `install.js` | Reemplazar `torch.js` cu128 por instalación de `mlx` o `ai-edge-litert` |

## Notas de rendimiento (desde las model cards)

- **LiteRT en Galaxy S26 GPU**: ~2.4–2.5 s por segundo de audio (más lento que real time)
- **LiteRT en Mac M4 Max GPU**: ~0.44–0.50 s por segundo de audio (más rápido que real time)
- **LiteRT en Mac M4 Max CPU**: ~3.0–3.2 s por segundo de audio
- **Load time en Mac GPU**: ~131–137 s (compilación del grafo LiteRT)
- **MLX en Apple Silicon**: pure-MLX, no requiere kernels CUDA, parity 0.9998 vs torch reference

## Archivos de referencia ya descargados

Ninguno de estos modelos ha sido descargado al repositorio. Solo se han capturado las URL y metadatos en este documento para referencia futura.

## Próximos pasos (futura fase Mac)

1. Crear una nueva fase (Phase 3) dedicada al soporte Mac/Apple Silicon.
2. Añadir `mlx` o `ai-edge-litert` a `requirements_full.txt` condicionalmente para `platform == "darwin"`.
3. Crear una rama de código separada para el backend MLX/LiteRT.
4. Añadir las variantes de modelo MLX al `VARIANT_REGISTRY` con `weight_source: "mlx"` o `weight_source: "litert"`.
5. Actualizar `pinokio.json` con `"platform": ["darwin"]` y `"arch": ["arm64"]`.

---

*Fecha: 2026-11-12*
*Fase: 02-enable-realtime-reuse-variants*