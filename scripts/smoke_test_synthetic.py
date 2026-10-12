#!/usr/bin/env python3
"""Run an explicitly synthetic, temporary end-to-end pipeline smoke test.

This fixture is software plumbing only. Its random pixels/captions do not
represent licensed footage or demonstrate useful video-generation quality.
Everything, including checkpoints, is deleted when the script exits.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))


def main() -> int:
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Install the project [training,preprocess] extras before running the synthetic smoke test."
        ) from exc

    from oasis_video.catalog import load_catalog, register_original_checkpoint
    from oasis_video.checkpoint import verify_checkpoint
    from oasis_video.evaluation import evaluate_checkpoint
    from oasis_video.inference import generate_video, load_verified_model
    from oasis_video.io_utils import sha256_file, write_json_atomic, write_jsonl_atomic
    from oasis_video.splits import make_splits
    from oasis_video.training import train_from_config
    import oasis_video.training as training

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="oasis-synthetic-smoke-") as temporary:
        root = Path(temporary)
        (root / "frames").mkdir()
        write_json_atomic(
            root / "registry.json",
            {
                "schema_version": 1,
                "datasets": [
                    {
                        "dataset_id": "synthetic-smoke-fixture",
                        "name": "Procedurally generated test-only RGB patterns",
                        "source_url": "local:temporary-test-fixture",
                        "version": "1",
                        "acquired_at": "2026-10-12",
                        "rights_holder": "OASIS test fixture generator",
                        "license": {
                            "name": "Synthetic unit-test fixture; no external source material",
                            "url": "local:temporary-test-fixture",
                            "text_sha256": None,
                        },
                        "permissions": {
                            "training": "allowed",
                            "commercial_use": "unknown",
                            "redistribution": "unknown",
                        },
                        "attribution_required": False,
                        "attribution_text": "",
                        "privacy_restrictions": [],
                        "known_limitations": ["Random color arrays only; not representative video or captions"],
                        "status": "approved",
                        "review": {
                            "reviewed_by": "temporary smoke test",
                            "reviewed_at": "2026-10-12",
                            "evidence_uri": "local:temporary-generated-fixture",
                            "notes": "Temporary synthetic software-test fixture, not a real dataset rights review",
                        },
                        "integrity": {"source_manifest_sha256": None, "source_file_count": None},
                    }
                ],
            },
        )

        rows = []
        random = np.random.default_rng(2026)
        for clip_index in range(20):
            frame_dir = root / "frames" / f"clip-{clip_index:02d}"
            frame_dir.mkdir()
            frames = []
            for frame_index in range(8):
                pixels = random.integers(0, 256, (16, 16, 3), dtype=np.uint8)
                channel = (clip_index + frame_index) % 3
                pixels[:, :, channel] = (
                    (pixels[:, :, channel].astype(np.uint16) + clip_index * 7 + frame_index * 11) % 256
                ).astype(np.uint8)
                frame_path = frame_dir / f"frame-{frame_index:02d}.png"
                Image.fromarray(pixels, "RGB").save(frame_path)
                frames.append({"path": frame_path.relative_to(root).as_posix(), "sha256": sha256_file(frame_path)})
            rows.append(
                {
                    "item_id": f"synthetic-{clip_index:02d}",
                    "source_id": f"synthetic-source-{clip_index:02d}",
                    "dataset_id": "synthetic-smoke-fixture",
                    "dataset_version": "1",
                    "source_path": f"generated/source-{clip_index:02d}.mp4",
                    "source_sha256": hashlib.sha256(f"source-{clip_index}".encode()).hexdigest(),
                    "caption": "Procedural colored shapes move through a simple synthetic scene.",
                    "caption_reviewed": True,
                    "frame_count": 8,
                    "width": 16,
                    "height": 16,
                    "fps": 4.0,
                    "duration_seconds": 2.0,
                    "start_seconds": 0.0,
                    "data_root": ".",
                    "frame_paths": frames,
                    "visual_fingerprints": [],
                }
            )
        write_jsonl_atomic(root / "prepared.jsonl", rows)
        split_manifest, split_counts = make_splits(
            root / "prepared.jsonl",
            root / "registry.json",
            root / "splits.jsonl",
            seed=2026,
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15,
            near_duplicate_threshold=0,
        )
        if not all(split_counts[name] for name in ("train", "validation", "test")):
            raise RuntimeError(f"Synthetic split did not populate every partition: {split_counts}")

        config_path = root / "experiment.toml"
        config_path.write_text(
            '''[experiment]
name = "synthetic-pipeline-smoke"
seed = 20261012
output_dir = "runs"

[dataset]
registry = "registry.json"
manifest = "splits.jsonl"
train_split = "train"
validation_split = "validation"

[model]
frames = 8
width = 16
height = 16
channels = 3
base_channels = 8
condition_dim = 16
timesteps = 4
noise_schedule = "linear"
beta_start = 0.0001
beta_end = 0.02
max_text_bytes = 32

[training]
max_steps = 2
batch_size = 1
gradient_accumulation = 1
objective = "epsilon_mse"
optimizer = "AdamW"
learning_rate = 0.0001
learning_rate_schedule = "constant"
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.999
adam_epsilon = 0.00000001
max_grad_norm = 1.0
precision = "fp32"
checkpoint_every = 1
validation_every = 1
validation_batches = 1
deterministic = true
device = "cpu"
''',
            encoding="utf-8",
        )

        append_event = training._append_event

        def interrupt_after_step_one(path: Path, payload: dict) -> None:
            append_event(path, payload)
            if payload.get("event") == "checkpoint_saved" and payload.get("step") == 1:
                raise KeyboardInterrupt()

        with patch.object(training, "_append_event", side_effect=interrupt_after_step_one):
            interrupted_checkpoint = train_from_config(config_path)
        interrupted_metadata = verify_checkpoint(interrupted_checkpoint, require_resume_state=True)
        final_checkpoint = train_from_config(
            config_path,
            resume_checkpoint=interrupted_checkpoint,
            device_override="cpu",
        )
        final_metadata = verify_checkpoint(final_checkpoint, require_resume_state=True)

        catalog_path = root / "models" / "catalog.json"
        write_json_atomic(catalog_path, {"schema_version": 1, "models": []})
        register_original_checkpoint(final_checkpoint, catalog_path)
        if len(load_catalog(catalog_path)["models"]) != 1:
            raise RuntimeError("Temporary smoke checkpoint did not pass catalog validation")

        model, _, device = load_verified_model(final_checkpoint, device="cpu")
        generated = generate_video(model, "A temporary synthetic test prompt.", seed=99, device=device)
        if generated.shape != (8, 16, 16, 3) or generated.dtype != np.uint8:
            raise RuntimeError(f"Unexpected generated frame array: shape={generated.shape}, dtype={generated.dtype}")
        evaluation_path = evaluate_checkpoint(
            final_checkpoint,
            split_manifest,
            root / "registry.json",
            root / "evaluation",
            sample_count=1,
            seed=5,
            device="cpu",
        )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation["sample_count"] != 1:
            raise RuntimeError("Temporary evaluation did not complete")

        print(
            json.dumps(
                {
                    "result": "passed",
                    "synthetic_split_counts": split_counts,
                    "interrupted_checkpoint_step": interrupted_metadata["step"],
                    "resumed_checkpoint_step": final_metadata["step"],
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "generated_frame_array": list(generated.shape),
                    "held_out_evaluation_samples": evaluation["sample_count"],
                    "all_artifacts_temporary_and_deleted_on_exit": True,
                    "limitation": "Synthetic software smoke test only; it establishes no real-data training or video quality.",
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
