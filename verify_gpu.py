"""Verify that this install can actually run RE-USE on an NVIDIA GPU.

Checks CUDA availability, device identity, compute capability, the CUDA
architecture list, the torch family imports, and the mamba pure-Python path.

Prints a JSON summary and exits non-zero on failure. Safe to run repeatedly.
"""

import json
import sys


def main():
    result = {
        "ok": False,
        "torch": None,
        "torchvision": None,
        "torchaudio": None,
        "cuda_available": False,
        "device": None,
        "capability": None,
        "arch_list": None,
        "required_arch": ["sm_75", "sm_86"],
        "mamba_ssm": None,
        "mamba_patch_applied": False,
        "selective_scan_cuda": None,
        "selective_scan_ref": False,
        "selective_scan_bound": False,
        "errors": [],
    }

    try:
        import torch
    except Exception as exc:
        result["errors"].append("torch import failed: {}".format(exc))
        return finish(result)

    result["torch"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())

    if not result["cuda_available"]:
        result["errors"].append(
            "torch.cuda.is_available() is False. RE-USE requires an NVIDIA GPU "
            "with a CUDA 12.8-capable driver."
        )
        return finish(result)

    result["device"] = torch.cuda.get_device_name(0)
    result["capability"] = list(torch.cuda.get_device_capability(0))
    result["arch_list"] = list(torch.cuda.get_arch_list())

    for arch in result["required_arch"]:
        if arch not in result["arch_list"]:
            result["errors"].append(
                "PyTorch CUDA build is missing {}; the installed wheel cannot "
                "cover one of the supported GPU families.".format(arch)
            )

    for name in ("torchvision", "torchaudio"):
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            result["errors"].append("{} import failed: {}".format(name, exc))

    try:
        from reuse_runtime import (
            bind_selective_scan_reference,
            patch_mamba_selective_scan,
        )

        result["mamba_patch_applied"] = patch_mamba_selective_scan()
        if not result["mamba_patch_applied"]:
            result["errors"].append(
                "mamba_ssm is not installed (selective_scan_interface.py not found)."
            )
        else:
            import mamba_ssm
            import mamba_ssm.ops.selective_scan_interface as ssi

            result["mamba_ssm"] = getattr(mamba_ssm, "__version__", "unknown")
            result["selective_scan_cuda"] = (
                None if ssi.selective_scan_cuda is None else "compiled"
            )
            result["selective_scan_ref"] = callable(
                getattr(ssi, "selective_scan_ref", None)
            )
            if not result["selective_scan_ref"]:
                result["errors"].append(
                    "selective_scan_ref is not callable; the pure-Python fallback is missing."
                )

            result["selective_scan_bound"] = bind_selective_scan_reference()
            if not result["selective_scan_bound"]:
                result["errors"].append(
                    "Could not bind mamba_simple.selective_scan_fn to selective_scan_ref."
                )
    except Exception as exc:
        result["errors"].append("mamba_ssm check failed: {}".format(exc))

    return finish(result)


def finish(result):
    result["ok"] = not result["errors"]
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        sys.stderr.write("GPU verification failed:\n")
        for error in result["errors"]:
            sys.stderr.write("  - {}\n".format(error))
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(1)
    sys.exit(0)
