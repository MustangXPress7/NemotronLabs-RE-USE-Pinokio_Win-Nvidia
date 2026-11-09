"""Shared runtime compatibility helpers for the RE-USE launcher.

The pinned mamba revision (v2.3.2.post1) hard-imports the compiled
`selective_scan_cuda` extension. This launcher installs mamba as a pure-Python
package, so the import must be optional and the model must use mamba's reference
selective-scan implementation instead.

Both app.py and verify_gpu.py call these helpers, so the patch exists in exactly
one place and is applied at most once.
"""

import pathlib
import sysconfig


def patch_mamba_selective_scan():
    """Make mamba's hard `selective_scan_cuda` import optional.

    Returns True when the mamba module file exists, False otherwise. The rewrite
    itself is idempotent: a file that already has the guard is left untouched.
    """
    try:
        site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
        target = site_packages / "mamba_ssm" / "ops" / "selective_scan_interface.py"
        if not target.exists():
            return False

        source = target.read_text(encoding="utf-8")
        original = "import selective_scan_cuda\n"
        replacement = (
            "try:\n"
            "    import selective_scan_cuda\n"
            "except ImportError:\n"
            "    selective_scan_cuda = None\n"
        )

        if original in source and "selective_scan_cuda = None" not in source:
            target.write_text(source.replace(original, replacement, 1), encoding="utf-8")

        return True
    except Exception:
        return False


def bind_selective_scan_reference():
    """Route mamba_simple.selective_scan_fn to the pure-Python reference kernel.

    Returns True only when the assignment took effect. Raises if mamba_ssm
    cannot be imported, so callers fail loudly instead of running a broken model.
    """
    import mamba_ssm.ops.selective_scan_interface as ssi
    import mamba_ssm.modules.mamba_simple as mamba_simple

    mamba_simple.selective_scan_fn = ssi.selective_scan_ref
    return mamba_simple.selective_scan_fn is ssi.selective_scan_ref
