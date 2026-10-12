"""Repeatable held-out denoising evaluation and generated-sample review packs."""

from __future__ import annotations

import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import VideoClipDataset
from .inference import generate_video, load_verified_model
from .io_utils import ValidationError, read_json, sha256_file, write_json_atomic
from .manifests import validate_manifest
from .provenance import load_registry


def evaluate_checkpoint(
    checkpoint: str | Path,
    manifest_path: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    sample_count: int = 8,
    seed: int = 271828,
    device: str = "auto",
) -> Path:
    if split != "test":
        raise ValidationError("release evaluation defaults to the held-out 'test' split; training/validation leakage is not accepted")
    if sample_count < 1:
        raise ValidationError("sample_count must be at least 1")
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Evaluation requires numpy, Pillow, and PyTorch") from exc

    registry = load_registry(registry_path)
    split_meta_path = Path(str(manifest_path) + ".meta.json")
    if not split_meta_path.is_file():
        raise ValidationError("Evaluation requires split metadata from oasis-research make-splits")
    split_metadata = read_json(split_meta_path)
    if split_metadata.get("output_manifest_sha256") != sha256_file(manifest_path):
        raise ValidationError("Evaluation split metadata does not match the manifest checksum")
    if split_metadata.get("registry_sha256") != sha256_file(registry_path):
        raise ValidationError("Evaluation split metadata does not match the rights registry checksum")
    full_rows = validate_manifest(
        manifest_path, registry, require_prepared=True, for_training=True, split="test", verify_frame_hashes=True
    )
    count = min(sample_count, len(full_rows))
    if count < sample_count:
        note = f"Requested {sample_count} samples; only {count} held-out test clips are available."
    else:
        note = None
    model, checkpoint_metadata, selected_device = load_verified_model(checkpoint, device=device)
    config = checkpoint_metadata["model_config"]
    dataset = VideoClipDataset(
        manifest_path,
        registry_path,
        split="test",
        frames=config["frames"],
        width=config["width"],
        height=config["height"],
        max_text_bytes=config["max_text_bytes"],
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples_dir = output / "samples"
    samples_dir.mkdir()
    review_rows: list[dict[str, Any]] = []
    heldout_losses: list[float] = []
    temporal_deltas: list[float] = []
    exact_duplicate_rates: list[float] = []
    betas = torch.linspace(
        config["beta_start"], config["beta_end"], config["timesteps"], device=selected_device, dtype=torch.float32
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    for index in range(count):
        video, token_ids, caption = dataset[index]
        video = video.unsqueeze(0).to(selected_device)
        token_ids = token_ids.unsqueeze(0).to(selected_device)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed + index)
        time_index = torch.randint(config["timesteps"], (1,), generator=rng, device="cpu").to(selected_device)
        noise = torch.randn(video.shape, generator=rng, device="cpu").to(selected_device)
        alpha_bar = alpha_bars[time_index].view(1, 1, 1, 1, 1)
        noisy = alpha_bar.sqrt() * video + (1.0 - alpha_bar).sqrt() * noise
        with torch.inference_mode():
            predicted = model(noisy, time_index, token_ids)
        heldout_loss = torch.mean((predicted.float() - noise) ** 2)
        if not torch.isfinite(heldout_loss).item():
            raise ValidationError(f"Non-finite held-out loss for sample {index}")
        heldout_losses.append(float(heldout_loss.cpu().item()))

        generated = generate_video(model, caption, seed=seed + 10_000 + index, device=selected_device)
        sample_folder = samples_dir / f"sample-{index:04d}"
        sample_folder.mkdir()
        for frame_index, frame in enumerate(generated):
            Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(
                sample_folder / f"frame-{frame_index:04d}.png"
            )
        if len(generated) > 1:
            differences = [
                float(np.abs(generated[frame].astype(np.float32) - generated[frame - 1].astype(np.float32)).mean())
                for frame in range(1, len(generated))
            ]
            temporal_deltas.append(statistics.mean(differences))
            exact_duplicate_rates.append(
                sum(int(np.array_equal(generated[frame], generated[frame - 1])) for frame in range(1, len(generated)))
                / (len(generated) - 1)
            )
        review_rows.append(
            {
                "sample_id": f"sample-{index:04d}",
                "prompt": caption,
                "seed": seed + 10_000 + index,
                "human_visual_quality_1_to_5": "",
                "human_temporal_coherence_1_to_5": "",
                "human_prompt_adherence_1_to_5": "",
                "reviewer_notes": "",
            }
        )

    with (output / "human-review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    report = {
        "schema_version": 1,
        "evaluation_type": "held_out_test_research_preview",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": checkpoint_metadata.get("model_version"),
        "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
        "architecture_id": checkpoint_metadata["architecture_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "split_metadata_sha256": sha256_file(split_meta_path),
        "registry_sha256": sha256_file(registry_path),
        "split": "test",
        "sample_count": count,
        "requested_sample_count": sample_count,
        "seed": seed,
        "device": selected_device,
        "heldout_epsilon_mse": statistics.mean(heldout_losses),
        "heldout_epsilon_mse_per_sample": heldout_losses,
        "sample_diagnostics": {
            "mean_adjacent_frame_pixel_l1_0_to_255": statistics.mean(temporal_deltas) if temporal_deltas else None,
            "mean_exact_duplicate_adjacent_frame_rate": statistics.mean(exact_duplicate_rates) if exact_duplicate_rates else None,
            "interpretation": "Basic pixel diagnostics only; they do not measure semantic motion quality or prompt adherence.",
        },
        "limitations": [
            "Held-out epsilon MSE is a training-objective diagnostic, not a video-quality score.",
            "No pretrained video/text encoder or FVD implementation is included; no prompt-adherence score is fabricated.",
            "Review generated PNG sequences and complete human-review.csv before making quality claims.",
        ],
        "note": note,
        "human_review_csv": "human-review.csv",
        "samples_directory": "samples/",
    }
    report_path = output / "evaluation.json"
    write_json_atomic(report_path, report)
    return report_path
