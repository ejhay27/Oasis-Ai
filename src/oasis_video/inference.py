"""Safe inference adapter for checkpoints produced by the OASIS research trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import verify_checkpoint
from .io_utils import ValidationError
from .model import beta_schedule, create_model, encode_caption_ids


def choose_device(requested: str = "auto") -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for inference; install the 'training' extra") from exc
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ValidationError("CUDA was requested but is not available in this PyTorch installation")
        if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ValidationError("MPS was requested but is not available in this PyTorch installation")
        if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
            raise ValidationError(f"Unsupported inference device {requested!r}")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_verified_model(checkpoint: str | Path, *, device: str = "auto") -> tuple[Any, dict[str, Any], str]:
    """Load only the fixed OASIS architecture from a hash-verified safetensors file."""
    metadata = verify_checkpoint(checkpoint)
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("PyTorch and safetensors are required for inference") from exc
    selected_device = choose_device(device)
    model = create_model(metadata["model_config"])
    metadata_path = Path(checkpoint)
    if metadata_path.suffix.lower() != ".json":
        metadata_path = metadata_path.with_suffix(".json")
    weights_path = metadata_path.resolve().parent / metadata["checkpoint_file"]
    weights = load_file(str(weights_path), device="cpu")
    state = {key.removeprefix("model."): value for key, value in weights.items() if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.to(selected_device)
    model.eval()
    return model, metadata, selected_device


def generate_video(
    model: Any,
    caption: str,
    *,
    seed: int,
    device: str,
) -> Any:
    """Sample one fixed-shape text-conditioned RGB clip with ancestral DDPM sampling.

    Returns uint8 frames with layout [T,H,W,C]. This baseline sampler is
    intentionally simple and is not an optimized production inference path.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for inference") from exc
    config = {
        "frames": model.frames,
        "height": model.height,
        "width": model.width,
        "channels": model.channels,
        "timesteps": model.timesteps,
        "beta_start": model.beta_start,
        "beta_end": model.beta_end,
        "max_text_bytes": model.max_text_bytes,
    }
    device_obj = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    betas = beta_schedule(
        model.timesteps,
        device=device_obj,
        dtype=torch.float32,
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    tokens = torch.tensor(
        [encode_caption_ids(caption, model.max_text_bytes)], dtype=torch.long, device=device_obj
    )
    sample = torch.randn(
        (1, config["channels"], config["frames"], config["height"], config["width"]),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device_obj)
    with torch.inference_mode():
        for step in reversed(range(model.timesteps)):
            timesteps = torch.full((1,), step, dtype=torch.long, device=device_obj)
            predicted_noise = model(sample, timesteps, tokens)
            alpha = alphas[step]
            beta = betas[step]
            alpha_bar = alpha_bars[step]
            mean = (sample - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if step > 0:
                previous_alpha_bar = alpha_bars[step - 1]
                variance = beta * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
                noise = torch.randn(sample.shape, generator=generator, device="cpu", dtype=torch.float32).to(device_obj)
                sample = mean + torch.sqrt(variance.clamp(min=1e-20)) * noise
            else:
                sample = mean
    if not torch.isfinite(sample).all().item():
        raise ValidationError("Sampler produced non-finite values; no frames were written")
    frames = sample[0].permute(1, 2, 3, 0).clamp(-1.0, 1.0).add(1.0).mul(127.5).round()
    return frames.to(device="cpu", dtype=torch.uint8).numpy()
