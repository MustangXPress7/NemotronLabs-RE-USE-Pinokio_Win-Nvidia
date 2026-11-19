"""Install verification for the Apple Silicon / LiteRT build.

Checks the host, the downloaded graphs and the ai-edge-litert import without
loading a graph (loading the main graph can take over two minutes).
"""

from __future__ import annotations

import json
import os
import platform

from mac_backend import MODEL_DIR, MODEL_SPECS, model_path

report = {
    "system": platform.system(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "model_dir": MODEL_DIR,
    "graphs": {},
    "checks": {},
}

report["checks"]["darwin"] = platform.system() == "Darwin"
report["checks"]["arm64"] = platform.machine() == "arm64"

try:
    import ai_edge_litert.compiled_model as compiled_model

    report["checks"]["ai_edge_litert_import"] = True
    report["compiled_model_api"] = "CompiledModel" in dir(compiled_model)
except Exception as exc:
    report["checks"]["ai_edge_litert_import"] = False
    report["import_error"] = f"{type(exc).__name__}: {exc}"

for key, spec in MODEL_SPECS.items():
    for graph, index in (("fastload", 1), ("main", 2)):
        filename = spec[index]
        path = os.path.join(MODEL_DIR, filename)
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        report["graphs"][f"{key}/{graph}"] = {"file": filename, "bytes": size, "ok": size > 0}

report["ok"] = (
    report["checks"]["darwin"]
    and report["checks"]["arm64"]
    and report["checks"]["ai_edge_litert_import"]
    and all(item["ok"] for item in report["graphs"].values())
)

print(json.dumps(report, indent=2, ensure_ascii=False))

if not report["ok"]:
    raise SystemExit(
        "Verification failed. Check that you are on an Apple Silicon Mac and that "
        "Install downloaded every LiteRT graph into models_litert/."
    )
