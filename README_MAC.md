# RE-USE Speech Enhancement · Apple Silicon (`mac-mlx`)

Launcher de Pinokio para mejorar grabaciones de voz en **Mac con Apple Silicon**.
Es la contrapartida de la rama `main` (Windows/Nvidia + CUDA). Ambas ramas viven
en el mismo repositorio git; usa `git merge main` desde esta rama para traer
arreglos compartidos.

## Qué hace

Restauración de voz **offline** con los grafos LiteRT de NVIDIA RE-USE
convertidos por la comunidad `litert-community`. El grafo corre en la GPU de
Apple vía el delegate Metal de LiteRT; la STFT, la iSTFT y la suma de chunks se
hacen en el host con NumPy.

- Punto de entrada único: `run.js` → `python app_mac.py`.
- Variantes: `Standard` y `Laughter` (preserva risas).
- Grafos: `fastload` (carga ~10 s) y `main` (carga ~131 s, referencia de calidad).
- Entrada: cualquier formato legible por `soundfile`, remuestreado a **16 kHz mono**.
- Salida: WAV float32 en `output/`, con log y espectrograma.
- No es un filtro de micro en tiempo real: los modelos disponibles en Mac son
  offline por archivo.

### Backends

| Backend | Estado | Rendimiento medido |
|---|---|---|
| LiteRT → Metal (Python) | **Activo**, backend por defecto | M4 Max: ~0,45 s por segundo de audio; M2 sin medir |
| MLX + Core ML (ReuseKit, Swift) | **Preparado**, no conectado a la UI | M2 Air: RTF 0,198 (MLX) / 0,147 (híbrido) |

## Instalación y uso en Pinokio

1. Añade esta carpeta como app (File → Add App → Select this folder).
2. **Install**: crea `.env` (Python 3.12), instala `requirements_mac.txt` con `uv`,
   descarga los cuatro grafos a `models_litert/` con `hf.download` y ejecuta
   `verify_mac.py`.
3. **Run RE-USE**: inicia Gradio; Pinokio captura la URL local y añade la pestaña
   *Open UI*.
4. **Verify Metal**: vuelve a comprobar host, archivos e importación.
5. **Update** / **Reset**: `git pull`, o borra `.env`, `cache` y los modelos.

Requisitos: macOS en `arm64`. `pinokio.json` declara `platform: darwin` y
`arch: arm64`, y `install.js` aborta antes de descargar si el host no coincide.

## Modelos y licencias

- https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT
- https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT
- https://huggingface.co/AILogDev/RE-USE-MLX-CoreML (fase 2)

Todos derivan de `nvidia/NVIDIA-NemotronLabs-RE-USE` y mantienen **OpenMDW-1.1**
(uso comercial permitido, con avisos de licencia/origen). El código de las
fichas de modelos es Apache-2.0.

## API

La app expone el endpoint de Gradio en la URL local que Pinokio guarda en la
variable `url` de `run.js`.

### JavaScript (Pinokio)

```javascript
// lanzar la app desde otro script
{ method: "script.start", params: { uri: "run.js" } }

// invocar el endpoint de Gradio (url = local.url de run.js)
const res = await fetch("{{local.url}}/run/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    data: [audioPath, "standard", "fastload", "gpu"],
    fn_index: 0
  })
});
const json = await res.json();
```

### Python

```python
from mac_backend import ReuseLiteRT, read_audio, write_wav

x, sr = read_audio("ruido.wav")
backend = ReuseLiteRT.load("standard", graph="fastload", accel="gpu")
result = backend.enhance(x)
write_wav("output/ruido_enhanced.wav", result.samples)
print(result.accel, result.seconds, result.chunks)
backend.close()
```

### Curl

```bash
curl -s -X POST "http://127.0.0.1:7860/run/predict" \
  -H "Content-Type: application/json" \
  -d '{"data":["/ruta/a/ruido.wav","standard","fastload","gpu"],"fn_index":0}'
```

El puerto lo elige Gradio; usa siempre la URL capturada por `run.js` en lugar
de hardcodear `7860`.

## Fase 2: backend MLX nativo

`mlx_backend.py` reserva la interfaz. Para activarlo:

```bash
hf download AILogDev/REUSE-MLX-CoreML --local-dir ModelPack
# extraer ReuseKit.zip junto a ModelPack
cd ReuseKit
xcrun swift run --build-system swiftbuild -c release reuse-probe \
  ../ModelPack input.wav output mlx 3
```

ReuseKit es un paquete Swift (`ReuseEngine.enhance`, modos `.mlx` y `.hybrid`),
no un módulo Python. Conectarlo a Gradio requiere un pequeño CLI Swift que lea
y escriba WAV; hasta entonces la UI solo ofrece LiteRT.
