# RE-USE Speech Enhancement · Mac Untested (`mac-mlx`)

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

## Cómo instalar esta rama (betatester)

**Importante:** el flujo *Create App → Download* de Pinokio 8.3.7 **no permite
elegir rama**: solo pide Git URL y nombre de carpeta, y clona con
`git clone --depth 1 --single-branch`. Si se pega solo la URL, se obtiene `main`
(Windows/CUDA). Tampoco funcionan sufijos como `#mac-mlx`, `/tree/mac-mlx` ni
`?branch=mac-mlx`: se pasan tal cual al clone.

### Forma recomendada: launcher puente (basta una URL)

Existe un repo delgado que clona esta rama automáticamente. En Pinokio →
**Create App → Download**, pegar:

```
https://github.com/MustangXPress7/RE-USE-AppleSilicon.git
```

Nombre de carpeta: `RE-USE-AppleSilicon`. Su `install.js` hace
`git clone --depth 1 --branch mac-mlx ... app` y luego ejecuta `app/install.js`.
Después: **Install** → **Run RE-USE** → pestaña *Open UI*.

### Alternativa: clonar la rama a mano

```bash
cd ~/.pinokio/api
git clone --depth 1 --branch mac-mlx https://github.com/MustangXPress7/NemotronLabs-RE-USE-Pinokio_Win-Nvidia.git RE-USE-MLX
```

(En macOS la ruta es `~/.pinokio/api`; en Windows sería `%USERPROFILE%\.pinokio\api`.)

En Pinokio → **Home** aparecerá la app **RE-USE Speech Enhancement (Apple
Silicon)**. Abrirla y seguir: **Install** → **Run RE-USE** → pestaña *Open UI*.

Si algo falla, adjunta el contenido de `logs/api/` (`install.js`, `run.js`,
`verify.js`) y `logs/shell/`, además de la salida de `python verify_mac.py`.

## Instalación y uso en Pinokio (carpeta local ya clonada)

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

`mlx_backend.py` reserva la interfaz. Hay cuatro caminos (cinco repositorios);
**ninguno está conectado a la UI todavía** y ninguno cambia el Install actual.

| Candidato | Tipo | Base | Licencia | requisito | RTF medido |
|---|---|---|---|---|---|
| `AILogDev/REUSE-MLX-CoreML` | MLX + Core ML (Swift/ReuseKit) | NemotronLabs RE-USE | **OpenMDW-1.1, uso comercial permitido** | Xcode 27.2 + Swift 6.4.2 | 0,198 / 0,147 (M2 Air) |
| `okdohyuk/RE-USE-MLX-CoreML` | **Espejo sin cambios** del anterior (rev. `4b7e506`), para Nogmi | NemotronLabs RE-USE | **OpenMDW-1.1, uso comercial permitido** | igual que AILogDev; `ReuseKit.zip` NO está en el espejo | ídem |
| `appautomaton/re-use-semamba-mlx` | **MLX puro en Python** (`mlx-speech`) | `nvidia/RE-USE` (anterior) | **NSCLv1, solo no comercial** | **Python 3.13+** | no publicado (paridad 0,9998) |
| `faraday/re-use-mlx` | MLX Swift (`model_mlx.safetensors`, 38,6 MB) | `nvidia/RE-USE` (rev. `7619050`) | **NSCLv1, solo no comercial** | app/CLI nativa [SpeechLens](https://github.com/faraday/SpeechLens) | no publicado |

### Opción A — ReuseKit (la recomendada si hay uso comercial)

```bash
hf download AILogDev/REUSE-MLX-CoreML --local-dir ModelPack
# extraer ReuseKit.zip junto a ModelPack
cd ReuseKit
xcrun swift run --build-system swiftbuild -c release reuse-probe \
  ../ModelPack input.wav output mlx 3
```

ReuseKit es un paquete Swift (`ReuseEngine.enhance`, modos `.mlx` y `.hybrid`),
no un módulo Python. Conectarlo a Gradio requiere un pequeño CLI Swift que lea
y escriba WAV.

### Opción B — `mlx-speech` (tentador, pero con tres bloqueos)

```bash
hf download appautomaton/re-use-semamba-mlx --local-dir models/reuse/mlx
python -c "from mlx_speech.generation.reuse import REUSEEnhancer; ..."
```

Es la única opción **100 % Python** (resuelve el problema del puente Swift), pero:

1. **Licencia:** deriva de `nvidia/RE-USE` bajo NSCLv1 (no comercial), mientras
   que los grafos LiteRT que ya usamos son OpenMDW-1.1 (comercial permitido).
   Meterlo en el mismo launcher mezclaría dos regímenes incompatibles.
2. **Python:** `mlx-speech` exige **3.13+**; nuestro `.env` es 3.12 porque
   `ai-edge-litert==2.3.0` se validó ahí. Haría falta un segundo entorno
   (p. ej. `.env-mlx`) o arriesgar el backend principal.
3. **Generación distinta:** es el RE-USE anterior, no NemotronLabs; no es
   equivalente a la variante `standard` y no ofrece `laughter`.

### Opción C — SpeechLens y el espejo `okdohyuk`

`faraday/re-use-mlx` trae pesos MLX Swift de 30 capas para la app/CLI nativa
**SpeechLens** (menu bar de macOS). Es la vía más parecida a un "CLI listo" de
las que hemos visto, pero su licencia NSCLv1 también es **solo no comercial** y
viene del `nvidia/RE-USE` antiguo. Acepta 8–48 kHz mono según su ficha.

`okdohyuk/RE-USE-MLX-CoreML` no aporta un modelo nuevo: es un **espejo byte a
byte** de `AILogDev/REUSE-MLX-CoreML` en la revisión fijada `4b7e506`
(38 archivos, 54.206.689 bytes, mismos hashes). Su utilidad es ser una URL
**estable y fijada** por si el repositorio original cambia; pero **no incluye
`ReuseKit.zip`**, que hay que seguir bajando de AILogDev. Licencia OpenMDW-1.1,
igual que el original.

Decisión pendiente: añadir una variante *legacy / no comercial* en un entorno
aparte, o quedarse con la vía OpenMDW-1.1 (AILogDev o su espejo) para la fase 2.
Hasta entonces, la UI solo ofrece LiteRT.
