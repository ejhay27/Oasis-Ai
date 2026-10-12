"""Command-line entry point. Training is never started by import or app launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .io_utils import ValidationError, write_json_atomic


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oasis-research",
        description="Local research and provenance tools for the OASIS video-model program.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    hardware = commands.add_parser("hardware", help="inspect local CPU, memory, storage, GPU, and framework availability")
    hardware.add_argument("--storage-path", default=".")

    registry = commands.add_parser("validate-registry", help="validate the dataset rights registry")
    registry.add_argument("path", type=Path)

    manifest = commands.add_parser("validate-manifest", help="validate a source or prepared JSONL manifest")
    manifest.add_argument("path", type=Path)
    manifest.add_argument("--registry", required=True, type=Path)
    manifest.add_argument("--prepared", action="store_true", help="require prepared frame sequences")
    manifest.add_argument("--for-training", action="store_true", help="also require reviewed captions and split metadata")
    manifest.add_argument("--split", choices=("train", "validation", "test"))
    manifest.add_argument("--skip-frame-hash-verification", action="store_true")

    inspect = commands.add_parser("inspect-video", help="probe codec, dimensions, frame rate, and duration with ffprobe")
    inspect.add_argument("path", type=Path)

    preprocess = commands.add_parser("preprocess", help="extract fixed-size PNG clips from an approved source manifest")
    preprocess.add_argument("--manifest", required=True, type=Path, help="one source video per JSONL row")
    preprocess.add_argument("--registry", required=True, type=Path)
    preprocess.add_argument("--output-dir", required=True, type=Path)
    preprocess.add_argument("--frames", type=int, default=8)
    preprocess.add_argument("--width", type=int, default=16)
    preprocess.add_argument("--height", type=int, default=16)
    preprocess.add_argument("--fps", type=float, default=4.0)
    preprocess.add_argument("--stride-seconds", type=float)
    preprocess.add_argument("--max-clips-per-source", type=int)

    splits = commands.add_parser("make-splits", help="assign immutable source/near-duplicate groups to train/val/test")
    splits.add_argument("--manifest", required=True, type=Path)
    splits.add_argument("--registry", required=True, type=Path)
    splits.add_argument("--output", required=True, type=Path)
    splits.add_argument("--seed", type=int, default=2026)
    splits.add_argument("--train-ratio", type=float, default=0.9)
    splits.add_argument("--validation-ratio", type=float, default=0.05)
    splits.add_argument("--test-ratio", type=float, default=0.05)
    splits.add_argument("--near-duplicate-threshold", type=int, default=4)

    config = commands.add_parser("validate-config", help="validate an experiment TOML configuration")
    config.add_argument("path", type=Path)

    train = commands.add_parser("train", help="explicitly start a local training experiment")
    train.add_argument("config", type=Path)
    train.add_argument("--resume", type=Path, help="resume from a verified OASIS training checkpoint")
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="override the versioned config device")

    verify = commands.add_parser("verify-checkpoint", help="verify checkpoint file hashes and metadata (does not execute code)")
    verify.add_argument("path", type=Path)

    register = commands.add_parser("register-checkpoint", help="register a trained OASIS checkpoint as experimental")
    register.add_argument("--checkpoint", required=True, type=Path)
    register.add_argument("--catalog", required=True, type=Path)

    catalog = commands.add_parser("validate-catalog", help="validate model categories and checkpoint hashes")
    catalog.add_argument("path", type=Path)

    generate = commands.add_parser("generate", help="generate a PNG-frame sequence from a verified OASIS checkpoint")
    generate.add_argument("--checkpoint", required=True, type=Path)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--seed", type=int, default=1234)
    generate.add_argument("--device", default="auto")

    evaluate = commands.add_parser("evaluate", help="run a fixed held-out test evaluation and create a human review pack")
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--registry", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--split", choices=("test",), default="test")
    evaluate.add_argument("--samples", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=271828)
    evaluate.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "hardware":
            from .hardware import inspect_hardware

            _json_print(inspect_hardware(args.storage_path))
        elif args.command == "validate-registry":
            from .provenance import load_registry

            datasets = load_registry(args.path)
            _json_print({"valid": True, "dataset_count": len(datasets), "registry": str(args.path.resolve())})
        elif args.command == "validate-manifest":
            from .manifests import validate_manifest
            from .provenance import load_registry

            registry = load_registry(args.registry)
            rows = validate_manifest(
                args.path,
                registry,
                require_prepared=args.prepared,
                for_training=args.for_training,
                split=args.split,
                verify_frame_hashes=not args.skip_frame_hash_verification,
            )
            counts: dict[str, int] = {}
            for row in rows:
                key = row.get("split", "unassigned")
                counts[key] = counts.get(key, 0) + 1
            _json_print({"valid": True, "row_count": len(rows), "split_counts": counts})
        elif args.command == "inspect-video":
            from .preprocess import inspect_video

            _json_print(inspect_video(args.path))
        elif args.command == "preprocess":
            from .preprocess import preprocess_manifest

            output = preprocess_manifest(
                args.manifest,
                args.registry,
                args.output_dir,
                frames=args.frames,
                width=args.width,
                height=args.height,
                fps=args.fps,
                stride_seconds=args.stride_seconds,
                max_clips_per_source=args.max_clips_per_source,
            )
            print(output)
        elif args.command == "make-splits":
            from .splits import make_splits

            output, counts = make_splits(
                args.manifest,
                args.registry,
                args.output,
                seed=args.seed,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
                near_duplicate_threshold=args.near_duplicate_threshold,
            )
            _json_print({"manifest": str(output), "counts": counts, "metadata": str(output) + ".meta.json"})
        elif args.command == "validate-config":
            from .config import load_config

            config = load_config(args.path)
            _json_print({
                "valid": True,
                "experiment": config.name,
                "model_shape": [config.model.frames, config.model.height, config.model.width],
                "manifest": str(config.manifest_path),
                "registry": str(config.registry_path),
                "device": config.training.device,
            })
        elif args.command == "train":
            from .training import train_from_config

            output = train_from_config(args.config, resume_checkpoint=args.resume, device_override=args.device)
            print(output)
        elif args.command == "verify-checkpoint":
            from .checkpoint import verify_checkpoint

            metadata = verify_checkpoint(args.path)
            _json_print({
                "valid": True,
                "model_version": metadata.get("model_version"),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "step": metadata.get("step"),
                "status": metadata.get("status"),
                "note": "Hash and metadata integrity only; not an independent legal, safety, or quality audit.",
            })
        elif args.command == "register-checkpoint":
            from .catalog import register_original_checkpoint

            record = register_original_checkpoint(args.checkpoint, args.catalog)
            _json_print(record)
        elif args.command == "validate-catalog":
            from .catalog import load_catalog

            catalog = load_catalog(args.path)
            _json_print({"valid": True, "model_count": len(catalog["models"])})
        elif args.command == "generate":
            from .inference import generate_video, load_verified_model
            import numpy as np
            from PIL import Image

            output = args.output_dir.resolve()
            model, metadata, device = load_verified_model(args.checkpoint, device=args.device)
            output.mkdir(parents=True, exist_ok=False)
            frames = generate_video(model, args.prompt, seed=args.seed, device=device)
            for index, frame in enumerate(frames):
                Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(output / f"frame-{index:04d}.png")
            write_json_atomic(
                output / "generation.json",
                {
                    "model_version": metadata.get("model_version"),
                    "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                    "prompt": args.prompt,
                    "seed": args.seed,
                    "device": device,
                    "frames": len(frames),
                    "output_format": "PNG frame sequence; video container encoding is not included",
                },
            )
            print(output)
        elif args.command == "evaluate":
            from .evaluation import evaluate_checkpoint

            report = evaluate_checkpoint(
                args.checkpoint,
                args.manifest,
                args.registry,
                args.output_dir,
                split=args.split,
                sample_count=args.samples,
                seed=args.seed,
                device=args.device,
            )
            print(report)
        return 0
    except (ValidationError, RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
