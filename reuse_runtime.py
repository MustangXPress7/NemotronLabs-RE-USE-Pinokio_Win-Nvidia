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


# Sampling below this many elements, the reference kernel amortises its Python
# loop over the sequence axis against a large batch. Above it, the reference is
# already fast and the chunked kernel's workspace would be needlessly large.
SCAN_CHUNKED_MAX_BATCH = 2048
SCAN_CHUNK_SIZE = 32


def selective_scan_chunked(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                           delta_softplus=False, return_last_state=False):
    """Selective scan solved in closed form over small chunks.

    Drop-in replacement for ``selective_scan_ref`` with the same signature. The
    reference walks the sequence axis in Python, which costs roughly one kernel
    launch per step. That is invisible when the batch is large (offline,
    whole-utterance inference) but dominates frame-by-frame streaming, where the
    batch is a single 20 ms frame and the sequence axis is the frequency axis.

    A first-order linear recurrence ``x_i = a_i * x_{i-1} + b_i`` with
    ``a_i = exp(delta_i * A)`` unrolls to

        x_i = sum_{j<=i} exp(cum_i - cum_j) * b_j + exp(cum_i) * x_0

    where ``cum`` is the running sum of ``log a``. Every exponent is non-positive
    (``delta > 0``, ``A < 0``), so the closed form cannot overflow; terms whose
    contribution has decayed away simply underflow to zero, as they should.

    Large batches, complex ``A`` and grouped ``B``/``C`` stay on the reference
    kernel, which is both correct and faster there.
    """
    import torch

    fallback = (
        A.is_complex()
        or B.dim() > 3
        or C.dim() > 3
        or B.dim() < 2
        or C.dim() < 2
        or u.shape[2] == 0
        or u.shape[0] * u.shape[1] > SCAN_CHUNKED_MAX_BATCH
    )
    if fallback:
        import mamba_ssm.ops.selective_scan_interface as ssi
        return ssi.selective_scan_ref(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
            delta_softplus=delta_softplus, return_last_state=return_last_state,
        )

    import torch.nn.functional as F

    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, length = u.shape
    A = A.float()
    static_B = B.dim() == 2
    static_C = C.dim() == 2
    B = B.float()
    C = C.float()
    x_prev = u.new_zeros((batch, dim, A.shape[1]))

    ys = []
    for start in range(0, length, SCAN_CHUNK_SIZE):
        end = min(start + SCAN_CHUNK_SIZE, length)
        span = end - start
        u_c = u[:, :, start:end]
        delta_c = delta[:, :, start:end]

        cum = torch.einsum("bdc,ds->bdcs", delta_c, A).cumsum(dim=2)
        if static_B:
            b_c = torch.einsum("bdc,ds,bdc->bdcs", delta_c, B, u_c)
        else:
            b_c = torch.einsum("bdc,bsc,bdc->bdcs", delta_c, B[:, :, start:end], u_c)

        # pairwise decay: weights[i, j] = exp(cum_i - cum_j) for j <= i
        mask = torch.triu(
            torch.ones(span, span, dtype=torch.bool, device=u.device), 1
        ).view(1, 1, span, span, 1)
        weights = (
            (cum.unsqueeze(3) - cum.unsqueeze(2))
            .masked_fill(mask, float("-inf"))
            .exp()
        )
        x = torch.einsum("bdijs,bdjs->bdis", weights, b_c)
        x = x + cum.exp() * x_prev.unsqueeze(2)
        x_prev = x[:, :, -1, :]
        del weights, b_c

        if static_C:
            ys.append(torch.einsum("bdcs,ds->bdc", x, C))
        else:
            ys.append(torch.einsum("bdcs,bsc->bdc", x, C[:, :, start:end]))

    y = torch.cat(ys, dim=2)
    last_state = x_prev

    out = y if D is None else y + u * D.float().reshape(1, -1, 1)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return (out, last_state) if return_last_state else out


def install_fast_selective_scan(verbose=False):
    """Bind the chunked selective scan when it matches the reference kernel.

    The self-test runs both kernels on random inputs shaped like a streaming
    frequency scan. Any mismatch (including NaN) leaves the reference kernel in
    place, so a numerical regression cannot silently change model behaviour.
    """
    import torch
    import mamba_ssm.ops.selective_scan_interface as ssi
    import mamba_ssm.modules.mamba_simple as mamba_simple

    if getattr(mamba_simple.selective_scan_fn, "_reuse_chunked", False):
        return True

    torch.manual_seed(0)
    batch, dim, length, n_state = 2, 32, 41, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u = torch.randn(batch, dim, length, device=device)
    delta = torch.rand(batch, dim, length, device=device) * 0.5 + 0.05
    A = -torch.exp(torch.randn(dim, n_state, device=device))
    B = torch.randn(batch, n_state, length, device=device) * 0.5
    C = torch.randn(batch, n_state, length, device=device) * 0.5
    D = torch.randn(dim, device=device)
    z = torch.randn(batch, dim, length, device=device)
    delta_bias = torch.randn(dim, device=device) * 0.1

    kwargs = dict(z=z, delta_bias=delta_bias, delta_softplus=True,
                  return_last_state=True)
    with torch.no_grad():
        try:
            ref_out, ref_state = ssi.selective_scan_ref(u, delta, A, B, C, D=D, **kwargs)
            new_out, new_state = selective_scan_chunked(u, delta, A, B, C, D=D, **kwargs)
        except Exception:
            return False

    if not (torch.isfinite(new_out).all() and torch.isfinite(new_state).all()):
        return False
    scale = ref_out.abs().max().clamp(min=1e-6)
    out_err = float((new_out - ref_out).abs().max() / scale)
    state_err = float((new_state - ref_state).abs().max() / scale)
    if out_err > 1e-3 or state_err > 1e-3:
        if verbose:
            print(f"[reuse] chunked scan rejected: out_err={out_err:.2e} state_err={state_err:.2e}")
        return False

    selective_scan_chunked._reuse_chunked = True
    mamba_simple.selective_scan_fn = selective_scan_chunked
    if verbose:
        print(f"[reuse] chunked selective scan installed (out_err={out_err:.2e})")
    return True
