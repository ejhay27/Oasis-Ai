"""Tiny OASIS spatiotemporal denoiser (research architecture v1).

This is an OASIS-authored compact pixel-space diffusion prototype. It uses
randomly initialized 3-D convolutions and a learned UTF-8 byte embedding; it
contains no pretrained encoder, VAE, teacher, or borrowed weights. Its existence
as code is not evidence that a trained model or useful generation quality exists.
"""

from __future__ import annotations

import math
from typing import Any


def encode_caption_ids(caption: str, max_text_bytes: int) -> list[int]:
    """Encode UTF-8 bytes as token IDs 1..256; zero is padding."""
    raw = caption.encode("utf-8")[:max_text_bytes]
    tokens = [byte + 1 for byte in raw]
    return tokens + [0] * (max_text_bytes - len(tokens))


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model training/inference; install the 'training' extra") from exc
    return torch


def _group_count(channels: int) -> int:
    return max(divisor for divisor in range(1, min(8, channels) + 1) if channels % divisor == 0)


def _build_classes() -> tuple[type, type]:
    torch = _require_torch()
    from torch import nn
    from torch.nn import functional as functional

    class ResidualBlock3D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
            super().__init__()
            self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.condition = nn.Linear(condition_dim, out_channels * 2)
            self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

        def forward(self, value: Any, condition: Any) -> Any:
            hidden = self.conv1(functional.silu(self.norm1(value)))
            scale, shift = self.condition(condition).chunk(2, dim=1)
            hidden = self.norm2(hidden)
            hidden = hidden * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
            hidden = self.conv2(functional.silu(hidden))
            return (self.skip(value) + hidden) / math.sqrt(2.0)

    class TinyVideoDenoiser(nn.Module):
        """Conditional epsilon-predicting 3D-convolutional denoiser.

        Tensor layout is B,C,T,H,W. Spatial downsampling is performed once;
        time is not downsampled in this deliberately small first experiment.
        """

        architecture_id = "oasis.tiny_video_diffusion.v1"

        def __init__(
            self,
            *,
            frames: int,
            width: int,
            height: int,
            channels: int = 3,
            base_channels: int = 32,
            condition_dim: int = 128,
            timesteps: int = 100,
            noise_schedule: str = "linear",
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            max_text_bytes: int = 96,
        ) -> None:
            super().__init__()
            if min(frames, width, height, channels, base_channels, condition_dim, timesteps, max_text_bytes) <= 0:
                raise ValueError("model dimensions and timestep count must be positive")
            if width < 8 or height < 8 or width % 2 or height % 2:
                raise ValueError("width and height must be >=8 and divisible by two")
            if noise_schedule != "linear" or not 0 < beta_start < beta_end < 1:
                raise ValueError("this architecture supports a linear beta schedule with 0 < beta_start < beta_end < 1")
            self.frames = frames
            self.width = width
            self.height = height
            self.channels = channels
            self.base_channels = base_channels
            self.condition_dim = condition_dim
            self.timesteps = timesteps
            self.noise_schedule = noise_schedule
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.max_text_bytes = max_text_bytes
            self.text_embedding = nn.Embedding(257, condition_dim, padding_idx=0)
            self.time_mlp = nn.Sequential(
                nn.Linear(condition_dim, condition_dim * 2),
                nn.SiLU(),
                nn.Linear(condition_dim * 2, condition_dim),
            )
            self.text_mlp = nn.Sequential(nn.Linear(condition_dim, condition_dim), nn.SiLU())

            self.input = nn.Conv3d(channels, base_channels, kernel_size=3, padding=1)
            self.encoder = ResidualBlock3D(base_channels, base_channels, condition_dim)
            self.downsample = nn.Conv3d(
                base_channels, base_channels * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
            )
            self.middle = ResidualBlock3D(base_channels * 2, base_channels * 2, condition_dim)
            self.upsample = nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1)
            self.decoder = ResidualBlock3D(base_channels * 2, base_channels, condition_dim)
            self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
            self.output = nn.Conv3d(base_channels, channels, kernel_size=3, padding=1)

        @staticmethod
        def _time_embedding(timesteps: Any, dimension: int) -> Any:
            half = dimension // 2
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=timesteps.device, dtype=torch.float32)
                / max(half - 1, 1)
            )
            phase = timesteps.float()[:, None] * frequencies[None, :]
            embedding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
            if dimension % 2:
                embedding = functional.pad(embedding, (0, 1))
            return embedding

        def forward(self, noisy_video: Any, timesteps: Any, token_ids: Any) -> Any:
            if noisy_video.ndim != 5:
                raise ValueError("noisy_video must have shape [batch, channels, frames, height, width]")
            if noisy_video.shape[1:] != (self.channels, self.frames, self.height, self.width):
                raise ValueError(
                    "input video shape does not match model config: expected "
                    f"[B,{self.channels},{self.frames},{self.height},{self.width}]"
                )
            mask = token_ids.ne(0).unsqueeze(-1)
            token_vectors = self.text_embedding(token_ids)
            text_vector = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            condition = self.time_mlp(self._time_embedding(timesteps, self.condition_dim)) + self.text_mlp(text_vector)

            skip = self.encoder(self.input(noisy_video), condition)
            hidden = self.middle(self.downsample(skip), condition)
            hidden = functional.interpolate(
                hidden,
                size=(self.frames, self.height, self.width),
                mode="trilinear",
                align_corners=False,
            )
            hidden = self.upsample(hidden)
            hidden = self.decoder(torch.cat((hidden, skip), dim=1), condition)
            return self.output(functional.silu(self.output_norm(hidden)))

    return TinyVideoDenoiser, nn.Module


def create_model(config: Any) -> Any:
    """Instantiate the fixed, code-defined architecture from a ModelConfig or mapping."""
    model_class, _ = _build_classes()
    if isinstance(config, dict):
        values = config
    else:
        values = vars(config)
    return model_class(
        frames=values["frames"],
        width=values["width"],
        height=values["height"],
        channels=values.get("channels", 3),
        base_channels=values["base_channels"],
        condition_dim=values["condition_dim"],
        timesteps=values["timesteps"],
        noise_schedule=values.get("noise_schedule", "linear"),
        beta_start=values.get("beta_start", 1e-4),
        beta_end=values.get("beta_end", 2e-2),
        max_text_bytes=values["max_text_bytes"],
    )


def beta_schedule(
    timesteps: int,
    *,
    device: Any,
    dtype: Any,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> Any:
    torch = _require_torch()
    if timesteps < 2 or not 0 < beta_start < beta_end < 1:
        raise ValueError("linear beta schedule requires timesteps>=2 and 0 < beta_start < beta_end < 1")
    return torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=dtype)
