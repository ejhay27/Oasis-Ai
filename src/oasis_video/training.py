"""Explicit, local-only training loop for the compact OASIS diffusion experiment."""

from __future__ import annotations

import json
import os
import platform
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

from . import __version__
from .checkpoint import restore_checkpoint, save_checkpoint, verify_checkpoint
from .config import ExperimentConfig, load_config
from .data import VideoClipDataset
from .hardware import inspect_hardware
from .io_utils import ValidationError, read_json, sha256_file, write_json_atomic
from .manifests import validate_manifest
from .model import beta_schedule, create_model
from .provenance import load_registry


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _distribution_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    return value[:64] or "experiment"


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment": {"name": config.name, "seed": config.seed, "output_dir": str(config.output_dir)},
        "dataset": {
            "registry": str(config.registry_path),
            "manifest": str(config.manifest_path),
            "train_split": config.train_split,
            "validation_split": config.validation_split,
        },
        "model": asdict(config.model),
        "training": asdict(config.training),
    }


def _resource_snapshot(device: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "system_memory_used_bytes": None,
        "process_rss_bytes": None,
        "gpu_memory_allocated_bytes": None,
        "gpu_memory_reserved_bytes": None,
        "gpu_utilization_percent": None,
    }
    try:
        import psutil

        snapshot["system_memory_used_bytes"] = int(psutil.virtual_memory().used)
        snapshot["process_rss_bytes"] = int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            snapshot["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
            snapshot["gpu_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
            try:
                import pynvml  # type: ignore[import-not-found]

                pynvml.nvmlInit()
                index = torch.device(device).index or 0
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                snapshot["gpu_utilization_percent"] = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception:
                pass
    except ImportError:
        pass
    return snapshot


def _encode_batch(dataset: VideoClipDataset, indices: list[int], device: Any) -> tuple[Any, Any]:
    import torch

    examples = [dataset[index] for index in indices]
    videos = torch.stack([example[0] for example in examples]).to(device)
    tokens = torch.stack([example[1] for example in examples]).to(device)
    return videos, tokens


def _autocast_context(device: Any, precision: str) -> Any:
    import torch

    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _calculate_loss(
    model: Any,
    videos: Any,
    tokens: Any,
    betas: Any,
    alpha_bars: Any,
    *,
    device: Any,
    precision: str,
    generator: Any | None = None,
) -> Any:
    import torch

    batch_size = videos.shape[0]
    if generator is None:
        timesteps = torch.randint(low=0, high=len(betas), size=(batch_size,), device=device)
        noise = torch.randn(videos.shape, device=device, dtype=torch.float32)
    else:
        # Use a CPU generator for fixed validation randomness on CPU, CUDA, and MPS.
        timesteps = torch.randint(low=0, high=len(betas), size=(batch_size,), device="cpu", generator=generator).to(device)
        noise = torch.randn(videos.shape, device="cpu", dtype=torch.float32, generator=generator).to(device)
    selected_alpha_bar = alpha_bars[timesteps].view(batch_size, 1, 1, 1, 1)
    noisy_video = selected_alpha_bar.sqrt() * videos + (1.0 - selected_alpha_bar).sqrt() * noise
    with _autocast_context(device, precision):
        predicted_noise = model(noisy_video, timesteps, tokens)
    loss = torch.mean((predicted_noise.float() - noise) ** 2)
    if not torch.isfinite(loss).item():
        raise ValidationError("Non-finite diffusion loss; preserving the last checkpoint and stopping this experiment")
    return loss


def _validate_run_inputs(config: ExperimentConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = load_registry(config.registry_path)
    all_rows = validate_manifest(
        config.manifest_path,
        registry,
        require_prepared=True,
        for_training=True,
        verify_frame_hashes=True,
    )
    split_meta_path = Path(str(config.manifest_path) + ".meta.json")
    if not split_meta_path.is_file():
        raise ValidationError("Training requires the split metadata sidecar produced by oasis-research make-splits")
    split_metadata = read_json(split_meta_path)
    if split_metadata.get("output_manifest_sha256") != sha256_file(config.manifest_path):
        raise ValidationError("Split metadata does not match the current manifest checksum")
    if split_metadata.get("registry_sha256") != sha256_file(config.registry_path):
        raise ValidationError("Split metadata does not match the current rights registry checksum")
    if split_metadata.get("output_manifest") != config.manifest_path.name:
        raise ValidationError("Split metadata refers to a different output manifest")
    train_rows = [row for row in all_rows if row.get("split") == config.train_split]
    validation_rows = [row for row in all_rows if row.get("split") == config.validation_split]
    test_rows = [row for row in all_rows if row.get("split") == "test"]
    if not train_rows:
        raise ValidationError("Manifest has no training rows; create group-aware splits with enough source videos")
    if not validation_rows:
        raise ValidationError("Manifest has no validation rows; create group-aware splits with enough source videos")
    if not test_rows:
        raise ValidationError("Manifest has no held-out test rows; create group-aware train/validation/test splits")
    for row in train_rows + validation_rows:
        if row["frame_count"] != config.model.frames or row["width"] != config.model.width or row["height"] != config.model.height:
            raise ValidationError(
                f"Clip {row['item_id']!r} shape is {row['frame_count']}x{row['height']}x{row['width']}; "
                f"config requires {config.model.frames}x{config.model.height}x{config.model.width}"
            )
    return train_rows, validation_rows


def _evaluate_validation(
    model: Any,
    dataset: VideoClipDataset,
    *,
    step: int,
    seed: int,
    batch_size: int,
    max_batches: int,
    betas: Any,
    alpha_bars: Any,
    device: Any,
    precision: str,
) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 50_000 + step)
    model.eval()
    losses: list[float] = []
    examples = min(len(dataset), batch_size * max_batches)
    with torch.inference_mode():
        for offset in range(0, examples, batch_size):
            indices = list(range(offset, min(offset + batch_size, examples)))
            videos, tokens = _encode_batch(dataset, indices, device)
            loss = _calculate_loss(
                model,
                videos,
                tokens,
                betas,
                alpha_bars,
                device=device,
                precision=precision,
                generator=generator,
            )
            losses.append(float(loss.cpu().item()))
    model.train()
    return {
        "validation_loss": sum(losses) / len(losses),
        "validation_batches": len(losses),
        "validation_examples": examples,
    }


def _make_grad_scaler(torch: Any, *, enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_from_config(
    config_path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    device_override: str | None = None,
) -> Path:
    """Run training only when this function is explicitly invoked by the CLI/API."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires PyTorch and numpy; install the 'training' extra") from exc

    config = load_config(config_path)
    train_rows, validation_rows = _validate_run_inputs(config)
    device_name = device_override or config.training.device
    if device_name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValidationError("device must be one of auto, cpu, cuda, mps")
    from .inference import choose_device

    device_name = choose_device("auto" if device_name == "auto" else device_name)
    device = torch.device(device_name)
    if config.training.precision != "fp32" and device.type != "cuda":
        raise ValidationError("fp16/bf16 training is currently supported only on CUDA; choose fp32 for CPU/MPS")
    if config.training.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValidationError("bf16 was requested but the selected CUDA device does not support it")

    config_hash = sha256_file(config.path)
    manifest_hash = sha256_file(config.manifest_path)
    registry_hash = sha256_file(config.registry_path)
    split_meta_path = Path(str(config.manifest_path) + ".meta.json")
    split_meta_hash = sha256_file(split_meta_path) if split_meta_path.is_file() else None
    hardware = inspect_hardware(config.output_dir)
    hardware["selected_device"] = str(device)
    hardware["training_precision"] = config.training.precision
    framework_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "numpy": np.__version__,
        "pillow": _distribution_version("Pillow"),
        "safetensors": _distribution_version("safetensors"),
        "oasis_video_research": __version__,
    }
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_dataset = VideoClipDataset(
        config.manifest_path,
        config.registry_path,
        split=config.train_split,
        frames=config.model.frames,
        width=config.model.width,
        height=config.model.height,
        max_text_bytes=config.model.max_text_bytes,
    )
    validation_dataset = VideoClipDataset(
        config.manifest_path,
        config.registry_path,
        split=config.validation_split,
        frames=config.model.frames,
        width=config.model.width,
        height=config.model.height,
        max_text_bytes=config.model.max_text_bytes,
    )
    if len(train_dataset) != len(train_rows) or len(validation_dataset) != len(validation_rows):
        raise ValidationError("Manifest row counts changed during dataset initialization")

    model = create_model(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.adam_epsilon,
    )
    scaler = _make_grad_scaler(torch, enabled=(device.type == "cuda" and config.training.precision == "fp16"))
    betas = beta_schedule(
        config.model.timesteps,
        device=device,
        dtype=torch.float32,
        beta_start=config.model.beta_start,
        beta_end=config.model.beta_end,
    )
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(config.seed + 1)

    run_id: str
    run_dir: Path
    run_json: Path
    events_path: Path
    step = epoch = cursor = 0
    order: list[int] = []
    elapsed_before = 0.0
    last_checkpoint: Path | None = None
    if resume_checkpoint is not None:
        resume_meta = verify_checkpoint(resume_checkpoint, require_resume_state=True)
        invariants = {
            "config_sha256": config_hash,
            "manifest_sha256": manifest_hash,
            "registry_sha256": registry_hash,
            "split_manifest_meta_sha256": split_meta_hash,
        }
        for key, expected in invariants.items():
            if resume_meta.get(key) != expected:
                raise ValidationError(f"Resume refused: checkpoint {key} does not match current inputs")
        if resume_meta.get("framework_versions") != framework_versions:
            raise ValidationError("Resume refused: Python/framework/library versions differ from checkpoint provenance")
        saved_hardware = resume_meta.get("hardware", {})
        for hardware_key in ("operating_system", "cpu_model", "nvidia_gpus"):
            if saved_hardware.get(hardware_key) != hardware.get(hardware_key):
                raise ValidationError(f"Resume refused: hardware field {hardware_key} differs from checkpoint provenance")
        if saved_hardware.get("selected_device") != str(device):
            raise ValidationError(
                f"Resume refused: checkpoint device {saved_hardware.get('selected_device')!r} differs from {str(device)!r}"
            )
        run_id = resume_meta.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValidationError("Resume checkpoint is missing run_id")
        run_dir = config.output_dir / run_id
        if not run_dir.is_dir():
            raise ValidationError(f"Resume run directory does not exist at expected path: {run_dir}")
        run_json = run_dir / "run.json"
        events_path = run_dir / "metrics.jsonl"
        try:
            previous_run = json.loads(run_json.read_text(encoding="utf-8"))
            elapsed_before = float(previous_run.get("elapsed_seconds") or 0.0)
        except (OSError, ValueError, TypeError):
            elapsed_before = 0.0
        restored_meta, progress = restore_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            order_generator=order_generator,
            scaler=scaler,
            map_location="cpu",
        )
        step = int(restored_meta["step"])
        if step >= config.training.max_steps:
            raise ValidationError("Checkpoint already reached configured max_steps; start a documented new run to extend training")
        epoch = int(progress["epoch"])
        cursor = int(progress["cursor"])
        order = [int(index) for index in progress["order"]]
        if sorted(order) != list(range(len(train_dataset))) or not 0 <= cursor <= len(order):
            raise ValidationError("Saved data-order state is incompatible with the current training manifest")
        last_checkpoint = Path(resume_checkpoint).resolve()
        _append_event(events_path, {"event": "resume", "timestamp": _utc_now(), "from_checkpoint": str(last_checkpoint), "step": step})
    else:
        run_id = f"{_slug(config.name)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-seed{config.seed}"
        run_dir = config.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run_json = run_dir / "run.json"
        events_path = run_dir / "metrics.jsonl"
        (run_dir / "checkpoints").mkdir()
        (run_dir / "config.toml").write_bytes(config.path.read_bytes())
        write_json_atomic(run_dir / "resolved-config.json", _resolved_config(config))

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "run_id": run_id,
        "run_directory": str(run_dir.resolve()),
        "config_path": str(config.path),
        "config_sha256": config_hash,
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": manifest_hash,
        "split_manifest_meta_path": str(split_meta_path) if split_meta_path.is_file() else None,
        "dataset_versions": [
            {"dataset_id": dataset_id, "version": dataset_version}
            for dataset_id, dataset_version in sorted(
                {(row["dataset_id"], row["dataset_version"]) for row in train_rows + validation_rows}
            )
        ],
        "registry_path": str(config.registry_path),
        "registry_sha256": registry_hash,
        "split_manifest_meta_sha256": split_meta_hash,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "training_configuration": _resolved_config(config),
        "framework_versions": framework_versions,
        "hardware": hardware,
    }
    if resume_checkpoint is None:
        write_json_atomic(run_json, {"run_id": run_id, "status": "running", "created_at": _utc_now(), **run_metadata})
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        _append_event(
            events_path,
            {
                "event": "run_started",
                "timestamp": _utc_now(),
                "step": 0,
                "train_examples": len(train_dataset),
                "validation_examples": len(validation_dataset),
                "parameter_count": parameter_count,
                "device": str(device),
                "precision": config.training.precision,
                "parameter_optimizer_bytes_lower_bound": parameter_count * 16,
                "memory_estimate_note": "Parameter/gradient/Adam-state lower bound only; activations/runtime overhead are excluded.",
            },
        )
    else:
        try:
            current_run = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current_run = {}
        current_run.update({"status": "running", "resumed_at": _utc_now(), "last_resumed_step": step})
        write_json_atomic(run_json, current_run)

    model.train()
    if not order:
        order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
        cursor = 0
    started_at = time.monotonic()
    step_started_at = started_at
    last_saved_step = step
    status = "completed"
    failure_message: str | None = None
    batch_size = min(config.training.batch_size, len(train_dataset))

    def save_at_current_step() -> Path:
        return save_checkpoint(
            checkpoint_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            model_config=asdict(config.model),
            run_metadata=run_metadata,
            progress={"step": step, "epoch": epoch, "cursor": cursor, "order": order},
            order_generator=order_generator,
            scaler=scaler,
        )

    try:
        while step < config.training.max_steps:
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            examples_this_step = 0
            for _ in range(config.training.gradient_accumulation):
                if cursor >= len(order):
                    epoch += 1
                    order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
                    cursor = 0
                indices = order[cursor : min(cursor + batch_size, len(order))]
                cursor += len(indices)
                if not indices:
                    raise ValidationError("Training sampler produced an empty batch")
                videos, tokens = _encode_batch(train_dataset, indices, device)
                loss = _calculate_loss(
                    model,
                    videos,
                    tokens,
                    betas,
                    alpha_bars,
                    device=device,
                    precision=config.training.precision,
                )
                scaler.scale(loss / config.training.gradient_accumulation).backward()
                micro_losses.append(float(loss.detach().float().cpu().item()))
                examples_this_step += len(indices)

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            if not torch.isfinite(gradient_norm).item():
                raise ValidationError("Non-finite gradient norm; preserving the last checkpoint and stopping this experiment")
            scaler.step(optimizer)
            scaler.update()
            step += 1
            step_elapsed = max(time.monotonic() - step_started_at, 1e-9)
            event: dict[str, Any] = {
                "event": "train_step",
                "timestamp": _utc_now(),
                "step": step,
                "epoch": epoch,
                "train_loss": sum(micro_losses) / len(micro_losses),
                "gradient_norm_before_clipping": float(gradient_norm.detach().float().cpu().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "examples_this_step": examples_this_step,
                "gradient_accumulation": config.training.gradient_accumulation,
                "step_seconds": step_elapsed,
                "examples_per_second": examples_this_step / step_elapsed,
                **_resource_snapshot(str(device)),
            }
            step_started_at = time.monotonic()
            if step % config.training.validation_every == 0 or step == config.training.max_steps:
                event.update(
                    _evaluate_validation(
                        model,
                        validation_dataset,
                        step=step,
                        seed=config.seed,
                        batch_size=batch_size,
                        max_batches=config.training.validation_batches,
                        betas=betas,
                        alpha_bars=alpha_bars,
                        device=device,
                        precision=config.training.precision,
                    )
                )
                event["event"] = "validation_step"
            _append_event(events_path, event)

            if step % config.training.checkpoint_every == 0 or step == config.training.max_steps:
                checkpoint_path = save_at_current_step()
                last_checkpoint = checkpoint_path
                last_saved_step = step
                _append_event(
                    events_path,
                    {"event": "checkpoint_saved", "timestamp": _utc_now(), "step": step, "metadata": str(checkpoint_path)},
                )
    except KeyboardInterrupt:
        status = "interrupted"
        _append_event(events_path, {"event": "controlled_interrupt", "timestamp": _utc_now(), "step": step})
    except Exception as exc:
        status = "failed"
        failure_message = f"{type(exc).__name__}: {exc}"
        out_of_memory = isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        if out_of_memory and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _append_event(
            events_path,
            {
                "event": "training_failure",
                "timestamp": _utc_now(),
                "step": step,
                "error": failure_message,
                "cuda_oom": out_of_memory,
                "recovery_guidance": (
                    "No experiment settings were changed. Consider a smaller resolution, shorter clips, smaller batch, "
                    "gradient accumulation, supported mixed precision, or a smaller model; record changes in a new config."
                    if out_of_memory
                    else None
                ),
            },
        )
        raise
    finally:
        if status == "interrupted" and step > last_saved_step:
            last_checkpoint = save_at_current_step()
            last_saved_step = step
            _append_event(
                events_path,
                {"event": "checkpoint_saved_after_interrupt", "timestamp": _utc_now(), "step": step, "metadata": str(last_checkpoint)},
            )
        write_json_atomic(
            run_json,
            {
                "run_id": run_id,
                "status": status,
                "updated_at": _utc_now(),
                "last_step": step,
                "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
                "elapsed_seconds": elapsed_before + time.monotonic() - started_at,
                "failure": failure_message,
                **run_metadata,
            },
        )

    return last_checkpoint or run_dir
