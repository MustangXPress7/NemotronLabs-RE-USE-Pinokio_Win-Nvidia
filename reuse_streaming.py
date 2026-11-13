# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Frame-by-frame streaming wrapper for the Real-time RE-USE model.

This is a port of ``online_inference.py`` from the official
``nvidia/Real-time_RE-USE`` model repo. The offline ``forward`` evaluates the
whole utterance in parallel, so every activation grows with the file length.
The streaming path keeps one 20 ms frame of context in cached causal-conv and
Mamba states, so memory is O(1) in the duration of the input.

The wrappers reuse the already-loaded weights by reference (no copy, no extra
parameters); the streaming model is just a different execution graph. The
attribute names mirror the offline model (``dense_encoder``, ``TSMamba``,
``mask_decoder_list``, ``phase_decoder_list``) so ``reuse_profiler`` can hook
the same modules in both modes.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal convolutions with a time cache
# ---------------------------------------------------------------------------
def get_causal_padding_2d(kernel_size, dilation=(1, 1)):
    """Causal (left-only) padding on the time axis, symmetric on frequency."""
    pad_t = kernel_size[0] * dilation[0] - dilation[0]
    pad_f = (kernel_size[1] * dilation[1] - dilation[1]) // 2
    return (pad_f, pad_f, pad_t, 0)  # (left, right, top, bottom)


class StreamingCausalConv2d(nn.Module):
    """Causal conv over time for ``[B, C, T, F]`` tensors, one frame at a time.

    The offline module prepends ``kernel_t - 1`` frames of zeros before the
    conv. Here those frames live in ``self.cache`` and are shifted forward on
    every call, which is what makes the conv causal and stateful.
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        kt, kf = conv.kernel_size
        dt, df = conv.dilation
        self.cache_len = (kt - 1) * dt
        self.register_buffer("cache", None, persistent=False)
        self.freq_pad = get_causal_padding_2d((1, kf), (dt, df))

    def reset(self):
        self.cache = None

    def forward(self, x, t_pad=None):
        # x: [B, C, 1, F]
        B, C, _, f = x.shape

        if self.cache is None:
            if t_pad is None:
                t_pad = self.cache_len
            self.cache = torch.zeros(
                B, C, t_pad, f, device=x.device, dtype=x.dtype
            )

        x_cat = torch.cat([self.cache, x], dim=2)
        y = self.conv(F.pad(x_cat, self.freq_pad, "constant", 0))
        self.cache = x_cat[:, :, -self.cache_len:, :]
        return y


class StreamingCausalConv2d_FT(nn.Module):
    """Time-cached conv for the decoder's transposed-conv stage.

    The decoder runs this conv on ``[B, C, F, T]`` (time last), so the cache is
    kept on the last axis instead of the third one.
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        kf, kt = conv.kernel_size
        df, dt = conv.dilation
        self.cache_len = (kt - 1) * dt
        self.register_buffer("cache", None, persistent=False)

    def reset(self):
        self.cache = None

    def forward(self, x):
        # x: [B, C, F, 1]
        B, C, f, _ = x.shape

        if self.cache is None:
            self.cache = torch.zeros(
                B, C, f, self.cache_len, device=x.device, dtype=x.dtype
            )
        x_cat = torch.cat([self.cache, x], dim=3)
        y = self.conv(x_cat)
        self.cache = x_cat[:, :, :, -self.cache_len:]
        return y


# ---------------------------------------------------------------------------
# Encoder / decoders
# ---------------------------------------------------------------------------
class StreamingDenseBlock(nn.Module):
    """Dense block where every conv keeps its own causal time cache."""

    def __init__(self, dense_block):
        super().__init__()
        self.blocks = nn.ModuleList()

        for seq in dense_block.dense_block:
            conv = seq[1]
            self.blocks.append(
                nn.Sequential(
                    StreamingCausalConv2d(conv),
                    seq[2],  # ChannelLayerNorm
                    seq[3],  # PReLU
                )
            )

    def reset(self):
        for block in self.blocks:
            block[0].reset()

    def forward(self, x):
        skip = x
        for block in self.blocks:
            y = block(skip)
            skip = torch.cat([y, skip], dim=1)
        return y


class StreamingDenseEncoder(nn.Module):
    """DenseEncoder with the look-ahead padding folded into the conv cache."""

    def __init__(self, encoder):
        super().__init__()

        conv1_1 = encoder.dense_conv_1_1[0]
        self.conv1_1 = nn.Sequential(
            StreamingCausalConv2d(conv1_1),
            encoder.dense_conv_1_1[1],
            encoder.dense_conv_1_1[2],
        )

        conv1_2 = encoder.dense_conv_1_2[0]
        self.conv1_2 = nn.Sequential(
            StreamingCausalConv2d(conv1_2),
            encoder.dense_conv_1_2[1],
            encoder.dense_conv_1_2[2],
        )

        conv1_3 = encoder.dense_conv_1_3[0]
        self.conv1_3 = nn.Sequential(
            StreamingCausalConv2d(conv1_3),
            encoder.dense_conv_1_3[1],
            encoder.dense_conv_1_3[2],
        )

        self.dense_block = StreamingDenseBlock(encoder.dense_block)

        # Time kernel is 1 here, so no cache is needed.
        self.conv2 = encoder.dense_conv_2

    def reset(self):
        self.conv1_1[0].reset()
        self.conv1_2[0].reset()
        self.conv1_3[0].reset()
        self.dense_block.reset()

    def forward(self, x, look_ahead_frames):
        # x: [B, C, 1, F]
        # The offline encoder pads +2 frames on the left and `look_ahead` on
        # the right. The cache holds the left part; the right part is the
        # caller's responsibility (it feeds look_ahead+1 frames on the first
        # call).
        if look_ahead_frames == 0:
            seq, pad = self.conv1_1, 2
        elif look_ahead_frames == 1:
            seq, pad = self.conv1_2, 1
        elif look_ahead_frames == 2:
            seq, pad = self.conv1_3, 0
        else:
            raise ValueError(
                f"look_ahead_frames must be 0, 1 or 2, got {look_ahead_frames}"
            )

        x = seq[0](x, t_pad=pad)
        for layer in seq[1:]:
            x = layer(x)

        x = self.dense_block(x)
        x = self.conv2(x)
        return x


class StreamingMagDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.up_conv1 = decoder.up_conv1
        self.up_conv2 = nn.Sequential(
            StreamingCausalConv2d_FT(decoder.up_conv2[0].conv),
            decoder.up_conv2[1],
            decoder.up_conv2[2],
        )
        self.final_conv = decoder.final_conv
        self.dense_block = StreamingDenseBlock(decoder.dense_block)

    def reset(self):
        self.dense_block.reset()
        self.up_conv2[0].reset()

    def forward(self, x):
        # x: [B, C, 1, F]
        x = self.dense_block(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.final_conv(x)
        return x


class StreamingPhaseDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.up_conv1 = decoder.up_conv1
        self.up_conv2 = nn.Sequential(
            StreamingCausalConv2d_FT(decoder.up_conv2[0].conv),
            decoder.up_conv2[1],
            decoder.up_conv2[2],
        )
        self.phase_conv_r = decoder.phase_conv_r
        self.phase_conv_i = decoder.phase_conv_i
        self.dense_block = StreamingDenseBlock(decoder.dense_block)

    def reset(self):
        self.dense_block.reset()
        self.up_conv2[0].reset()

    def forward(self, x):
        # x: [B, C, 1, F]
        x = self.dense_block(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


# ---------------------------------------------------------------------------
# Mamba stack
# ---------------------------------------------------------------------------
class StreamingCausalMambaBlock(nn.Module):
    """Causal (forward-only) Mamba block driven by ``Mamba.step()``.

    ``step`` keeps an O(1) conv/SSM state per frequency bin, so the temporal
    axis no longer contributes to memory.
    """

    def __init__(self, block):
        super().__init__()
        self.forward_blocks = block.forward_blocks
        self.output_proj = block.output_proj
        self.norm = block.norm
        self.register_buffer("conv_state", None, persistent=False)
        self.register_buffer("ssm_state", None, persistent=False)

    def reset(self, batch_size):
        self.conv_state, self.ssm_state = self.forward_blocks.allocate_inference_cache(
            batch_size=batch_size,  # number of frequency bins
            max_seqlen=1,
        )

    def forward(self, x):
        # x: [B, 1, D]
        y, self.conv_state, self.ssm_state = self.forward_blocks.step(
            x, self.conv_state, self.ssm_state
        )
        y = y + x
        y = self.output_proj(y)
        return self.norm(y)


class StreamingTFMambaBlock(nn.Module):
    """Temporal Mamba runs step-wise; frequency Mamba stays full-width."""

    def __init__(self, block):
        super().__init__()
        self.time_mamba = StreamingCausalMambaBlock(block.time_mamba)
        self.freq_mamba = block.freq_mamba  # frequency axis is always complete

    def reset(self, batch_size):
        self.time_mamba.reset(batch_size)

    def forward(self, x):
        # x: [B, C, 1, F]
        B, C, _, F = x.shape

        # ---- Temporal Mamba (true streaming) ----
        x_t = x.permute(0, 3, 2, 1).reshape(B * F, 1, C)
        x_t = self.time_mamba(x_t) + x_t
        x_t = x_t.view(B, F, 1, C).permute(0, 3, 2, 1)

        # ---- Frequency Mamba (non-causal, full frequency axis) ----
        x_f = x_t.permute(0, 2, 3, 1).reshape(B, F, C)
        x_f = self.freq_mamba(x_f) + x_f
        x_f = x_f.view(B, 1, F, C).permute(0, 3, 1, 2)

        return x_f


class StreamingSEMamba(nn.Module):
    """Streaming execution graph for ``SEMamba_decoder_list``.

    Weights are shared with the wrapped model; ``reset()`` must be called
    before every run (and after changing ``look_ahead_frames``).
    """

    def __init__(self, model):
        super().__init__()
        self.dense_encoder = StreamingDenseEncoder(model.dense_encoder)
        self.TSMamba = nn.ModuleList(
            [StreamingTFMambaBlock(block) for block in model.TSMamba]
        )
        self.mask_decoder_list = nn.ModuleList(
            [StreamingMagDecoder(decoder) for decoder in model.mask_decoder_list]
        )
        self.phase_decoder_list = nn.ModuleList(
            [StreamingPhaseDecoder(decoder) for decoder in model.phase_decoder_list]
        )

    def reset(self, freq_dim):
        self.dense_encoder.reset()
        for block in self.TSMamba:
            block.reset(freq_dim)
        for decoder in self.mask_decoder_list:
            decoder.reset()
        for decoder in self.phase_decoder_list:
            decoder.reset()

    def forward(self, noisy_mag_t, noisy_pha_t, layer_use, look_ahead_frames):
        """One streaming step.

        ``noisy_mag_t`` / ``noisy_pha_t``: ``[B, F, T_step]``, where
        ``T_step`` is ``look_ahead + 1`` on the first call and 1 afterwards.
        Returns ``(mag, pha)`` with shape ``[B, F]``.
        """
        x_mag = noisy_mag_t.permute(0, 2, 1).unsqueeze(1)  # [B, 1, T, F]
        x_pha = noisy_pha_t.permute(0, 2, 1).unsqueeze(1)  # [B, 1, T, F]
        x = torch.cat([x_mag, x_pha], dim=1)  # [B, 2, T, F]

        # Same 2-bin zero padding as the offline generator.
        zeros = torch.zeros(x.size(0), x.size(1), x.size(2), 2, device=x.device)
        x = torch.cat([x, zeros], dim=-1)

        x = self.dense_encoder(x, look_ahead_frames)

        for block in self.TSMamba[0:layer_use]:
            x = block(x)

        current_layer = layer_use - 1
        mag = self.mask_decoder_list[current_layer](x)
        pha = self.phase_decoder_list[current_layer](x)

        mag = mag.squeeze(2).squeeze(1)  # [B, F]
        pha = pha.squeeze(2).squeeze(1)  # [B, F]

        # The decoder can emit one extra bin; the offline model trims to F.
        mag = mag[:, 0 : x_mag.shape[-1]]
        pha = pha[:, 0 : x_mag.shape[-1]]
        return mag, pha


def build_streaming_model(model):
    """Wrap a loaded ``SEMamba_decoder_list`` for frame-by-frame inference."""
    return StreamingSEMamba(model).eval()


def run_streaming_frames(
    stream_model,
    noisy_mag,
    noisy_pha,
    exit_layer,
    look_ahead_frames=0,
    suppress_zero_mag=True,
):
    """Feed a full ``[B, F, T]`` spectrogram through ``stream_model`` one frame at a time.

    Mirrors the official ``online_inference.py`` loop: the first call consumes
    ``look_ahead + 1`` frames (to fill the look-ahead), every later call
    consumes exactly one, and ``look_ahead`` zero frames are appended so the
    last real frames still get their full context.

    Returns ``(mag, pha, frame_seconds)`` where ``mag``/``pha`` are
    ``[B, F, T]`` and ``frame_seconds`` is a list with the wall time of each
    step (measured with CUDA events when available).
    """
    look_ahead = int(look_ahead_frames or 0)
    n_frames = int(noisy_mag.shape[-1])

    if look_ahead:
        zeros_mag = torch.zeros(
            noisy_mag.shape[0], noisy_mag.shape[1], look_ahead,
            device=noisy_mag.device, dtype=noisy_mag.dtype,
        )
        zeros_pha = torch.zeros_like(zeros_mag)
        noisy_mag = torch.cat([noisy_mag, zeros_mag], dim=-1)
        noisy_pha = torch.cat([noisy_pha, zeros_pha], dim=-1)

    use_cuda = torch.cuda.is_available() and noisy_mag.is_cuda
    mag_out = []
    pha_out = []
    frame_seconds = []
    pending_events = []

    def _step(start, end):
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.perf_counter()

        mag_t, pha_t = stream_model(
            noisy_mag[:, :, start:end],
            noisy_pha[:, :, start:end],
            exit_layer,
            look_ahead,
        )

        if use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            # elapsed_time() only works once both events completed, so the
            # conversion happens after the loop.
            pending_events.append((start_event, end_event))
        else:
            frame_seconds.append(time.perf_counter() - t0)

        if suppress_zero_mag:
            # Official "strange sweep artifact" guard: frames that the model
            # zeroes out entirely are forced to zero in the output too.
            mag_check = torch.expm1(torch.relu(mag_t))
            zero_portion = (mag_check == 0).sum(dim=1) / mag_check.shape[1]
            if bool((zero_portion > 0.5).any()):
                mask = (zero_portion > 0.5).unsqueeze(1)
                mag_t = torch.where(mask, torch.zeros_like(mag_t), mag_t)

        mag_out.append(mag_t)
        pha_out.append(pha_t)

    # 1. Fill the look-ahead window.
    _step(0, look_ahead + 1)
    # 2. One frame in, one frame out.
    for t in range(look_ahead + 1, noisy_mag.shape[-1]):
        _step(t, t + 1)

    mag = torch.stack(mag_out, dim=-1)
    pha = torch.stack(pha_out, dim=-1)
    if use_cuda:
        torch.cuda.synchronize()
        for start_event, end_event in pending_events:
            frame_seconds.append(start_event.elapsed_time(end_event) / 1000.0)
    assert mag.shape[-1] == n_frames, (mag.shape, n_frames)
    return mag, pha, frame_seconds
