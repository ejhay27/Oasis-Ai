"""Version-controlled TOML experiment configuration loading and validation."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import ValidationError


@dataclass(frozen=True)
class ModelConfig:
    frames: int
    width: int
    height: int
    channels: int
    base_channels: int
    condition_dim: int
    timesteps: int
    noise_schedule: str
    beta_start: float
    beta_end: float
    max_text_bytes: int


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    batch_size: int
    gradient_accumulation: int
    objective: str
    optimizer: str
    learning_rate: float
    learning_rate_schedule: str
    weight_decay: float
    beta1: float
    beta2: float
    adam_epsilon: float
    max_grad_norm: float
    precision: str
    checkpoint_every: int
    validation_every: int
    validation_batches: int
    deterministic: bool
    device: str


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    name: str
    output_dir: Path
    seed: int
    registry_path: Path
    manifest_path: Path
    train_split: str
    validation_split: str
    model: ModelConfig
    training: TrainingConfig


def _object(payload: dict[str, Any], section: str) -> dict[str, Any]:
    value = payload.get(section)
    if not isinstance(value, dict):
        raise ValidationError(f"Configuration section [{section}] is required")
    return value


def _required(section: dict[str, Any], section_name: str, key: str, expected: type | tuple[type, ...]) -> Any:
    value = section.get(key)
    numeric_types = expected if isinstance(expected, tuple) else (expected,)
    is_numeric = any(kind in (int, float) for kind in numeric_types)
    if not isinstance(value, expected) or (is_numeric and isinstance(value, bool)):
        label = expected.__name__ if isinstance(expected, type) else " or ".join(kind.__name__ for kind in expected)
        raise ValidationError(f"[{section_name}].{key} must be {label}")
    return value


def _positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValidationError(f"{name} must be positive")


def _relative_path(value: str, config_dir: Path, name: str) -> Path:
    if not value.strip():
        raise ValidationError(f"{name} must not be empty")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError(f"Could not read TOML config {config_path}: {exc}") from exc

    exp = _object(raw, "experiment")
    dataset = _object(raw, "dataset")
    model = _object(raw, "model")
    training = _object(raw, "training")

    name = _required(exp, "experiment", "name", str)
    if not name.strip():
        raise ValidationError("[experiment].name must not be empty")
    seed = _required(exp, "experiment", "seed", int)
    if seed < 0 or seed > 2**32 - 1:
        raise ValidationError("[experiment].seed must be in [0, 2^32-1]")
    output_dir = _relative_path(_required(exp, "experiment", "output_dir", str), config_path.parent, "output_dir")
    registry_path = _relative_path(_required(dataset, "dataset", "registry", str), config_path.parent, "registry")
    manifest_path = _relative_path(_required(dataset, "dataset", "manifest", str), config_path.parent, "manifest")
    train_split = _required(dataset, "dataset", "train_split", str)
    validation_split = _required(dataset, "dataset", "validation_split", str)
    if train_split != "train" or validation_split != "validation":
        raise ValidationError("training must use the 'train' split and validation must use 'validation'")

    model_config = ModelConfig(
        frames=_required(model, "model", "frames", int),
        width=_required(model, "model", "width", int),
        height=_required(model, "model", "height", int),
        channels=_required(model, "model", "channels", int),
        base_channels=_required(model, "model", "base_channels", int),
        condition_dim=_required(model, "model", "condition_dim", int),
        timesteps=_required(model, "model", "timesteps", int),
        noise_schedule=_required(model, "model", "noise_schedule", str),
        beta_start=float(_required(model, "model", "beta_start", (int, float))),
        beta_end=float(_required(model, "model", "beta_end", (int, float))),
        max_text_bytes=_required(model, "model", "max_text_bytes", int),
    )
    for field_name in ("frames", "width", "height", "channels", "base_channels", "condition_dim", "timesteps", "max_text_bytes"):
        _positive(getattr(model_config, field_name), f"[model].{field_name}")
    if model_config.frames < 2:
        raise ValidationError("[model].frames must be at least 2")
    if model_config.width < 8 or model_config.height < 8:
        raise ValidationError("[model].width and height must be at least 8")
    if model_config.width % 2 or model_config.height % 2:
        raise ValidationError("[model].width and height must be divisible by 2")
    if model_config.channels != 3:
        raise ValidationError("the current experimental architecture accepts RGB video only (channels=3)")
    if model_config.base_channels < 8 or model_config.base_channels % 8:
        raise ValidationError("[model].base_channels must be >=8 and divisible by 8")
    if model_config.condition_dim < 16:
        raise ValidationError("[model].condition_dim must be >=16")
    if model_config.timesteps < 2:
        raise ValidationError("[model].timesteps must be >=2")
    if model_config.noise_schedule != "linear":
        raise ValidationError("the current model implementation supports noise_schedule='linear' only")
    if not (math.isfinite(model_config.beta_start) and math.isfinite(model_config.beta_end)):
        raise ValidationError("[model].beta_start and beta_end must be finite")
    if not 0 < model_config.beta_start < model_config.beta_end < 1:
        raise ValidationError("[model] requires 0 < beta_start < beta_end < 1")
    if model_config.max_text_bytes > 1024:
        raise ValidationError("[model].max_text_bytes must not exceed 1024")

    numeric = {
        key: float(_required(training, "training", key, (int, float)))
        for key in ("learning_rate", "weight_decay", "beta1", "beta2", "adam_epsilon", "max_grad_norm")
    }
    if not math.isfinite(numeric["learning_rate"]) or numeric["learning_rate"] <= 0:
        raise ValidationError("[training].learning_rate must be positive and finite")
    if not math.isfinite(numeric["weight_decay"]) or numeric["weight_decay"] < 0:
        raise ValidationError("[training].weight_decay must be non-negative and finite")
    if not 0 < numeric["beta1"] < 1 or not 0 < numeric["beta2"] < 1:
        raise ValidationError("[training].beta1 and beta2 must be between 0 and 1")
    if numeric["adam_epsilon"] <= 0 or not math.isfinite(numeric["adam_epsilon"]):
        raise ValidationError("[training].adam_epsilon must be positive and finite")
    if numeric["max_grad_norm"] <= 0 or not math.isfinite(numeric["max_grad_norm"]):
        raise ValidationError("[training].max_grad_norm must be positive and finite")
    training_config = TrainingConfig(
        max_steps=_required(training, "training", "max_steps", int),
        batch_size=_required(training, "training", "batch_size", int),
        gradient_accumulation=_required(training, "training", "gradient_accumulation", int),
        objective=_required(training, "training", "objective", str),
        optimizer=_required(training, "training", "optimizer", str),
        learning_rate=numeric["learning_rate"],
        learning_rate_schedule=_required(training, "training", "learning_rate_schedule", str),
        weight_decay=numeric["weight_decay"],
        beta1=numeric["beta1"],
        beta2=numeric["beta2"],
        adam_epsilon=numeric["adam_epsilon"],
        max_grad_norm=numeric["max_grad_norm"],
        precision=_required(training, "training", "precision", str),
        checkpoint_every=_required(training, "training", "checkpoint_every", int),
        validation_every=_required(training, "training", "validation_every", int),
        validation_batches=_required(training, "training", "validation_batches", int),
        deterministic=_required(training, "training", "deterministic", bool),
        device=_required(training, "training", "device", str),
    )
    for field_name in (
        "max_steps", "batch_size", "gradient_accumulation", "checkpoint_every", "validation_every", "validation_batches"
    ):
        _positive(getattr(training_config, field_name), f"[training].{field_name}")
    if training_config.objective != "epsilon_mse":
        raise ValidationError("the current training implementation supports objective='epsilon_mse' only")
    if training_config.optimizer != "AdamW":
        raise ValidationError("the current training implementation supports optimizer='AdamW' only")
    if training_config.learning_rate_schedule != "constant":
        raise ValidationError("the current training implementation supports learning_rate_schedule='constant' only")
    if training_config.precision not in {"fp32", "fp16", "bf16"}:
        raise ValidationError("[training].precision must be one of fp32, fp16, bf16")
    if training_config.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValidationError("[training].device must be one of auto, cpu, cuda, mps")
    return ExperimentConfig(
        path=config_path,
        name=name,
        output_dir=output_dir,
        seed=seed,
        registry_path=registry_path,
        manifest_path=manifest_path,
        train_split=train_split,
        validation_split=validation_split,
        model=model_config,
        training=training_config,
    )
