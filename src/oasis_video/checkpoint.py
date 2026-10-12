"""Safe checkpoint writing, integrity verification, and exact training resume.

Inference weights and optimizer/RNG tensors use safetensors; provenance and
scalar state use JSON. No checkpoint path executes imported model code.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import ValidationError, read_json, sha256_file, write_json_atomic

ARTIFACT_TYPE = "oasis.video-checkpoint.v1"
ARCHITECTURE_ID = "oasis.tiny_video_diffusion.v1"


def checkpoint_metadata_path(path: str | Path) -> Path:
    source = Path(path)
    if source.suffix.lower() == ".json":
        return source
    if source.name.endswith(".state.safetensors"):
        raise ValidationError("Pass the model .safetensors file or its .json sidecar, not the resume-state shard")
    return source.with_suffix(".json")


def verify_checkpoint(path: str | Path, *, require_resume_state: bool = False) -> dict[str, Any]:
    metadata_path = checkpoint_metadata_path(path).resolve()
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict) or metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ValidationError(f"{metadata_path}: unsupported or missing checkpoint artifact_type")
    model_name = metadata.get("checkpoint_file")
    expected_hash = metadata.get("checkpoint_sha256")
    if not isinstance(model_name, str) or not isinstance(expected_hash, str):
        raise ValidationError(f"{metadata_path}: checkpoint_file and checkpoint_sha256 are required")
    model_path = (metadata_path.parent / model_name).resolve()
    if model_path.parent != metadata_path.parent.resolve():
        raise ValidationError(f"{metadata_path}: checkpoint_file must be adjacent to the sidecar")
    if not model_path.is_file() or sha256_file(model_path) != expected_hash:
        raise ValidationError(f"Checkpoint file missing or SHA-256 mismatch: {model_path}")
    if metadata.get("architecture_id") != ARCHITECTURE_ID:
        raise ValidationError(f"Unsupported architecture: {metadata.get('architecture_id')!r}")
    if not isinstance(metadata.get("model_config"), dict):
        raise ValidationError(f"{metadata_path}: model_config must be an object")

    state_name = metadata.get("training_state_file")
    state_hash = metadata.get("training_state_sha256")
    if require_resume_state or state_name is not None:
        if not isinstance(state_name, str) or not isinstance(state_hash, str):
            if require_resume_state:
                raise ValidationError(f"{metadata_path}: resumable checkpoint has no optimizer/RNG state reference")
        else:
            state_path = (metadata_path.parent / state_name).resolve()
            if state_path.parent != metadata_path.parent.resolve():
                raise ValidationError(f"{metadata_path}: training_state_file must be adjacent to the sidecar")
            if not state_path.is_file() or sha256_file(state_path) != state_hash:
                raise ValidationError(f"Training-state file missing or SHA-256 mismatch: {state_path}")
    if require_resume_state and metadata.get("resume_supported") is not True:
        raise ValidationError(f"{metadata_path}: checkpoint is not marked resumable")
    return metadata


def _safe_tensor_writer(tensors: dict[str, Any], destination: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for checkpoint writing; install the 'training' extra") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        safe_tensors = {key: value.detach().to(device="cpu").contiguous() for key, value in tensors.items()}
        save_file(safe_tensors, temp_name)
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValidationError(f"Unsupported non-tensor optimizer state value: {type(value).__name__}")


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    model: Any,
    optimizer: Any,
    model_config: dict[str, Any],
    run_metadata: dict[str, Any],
    progress: dict[str, Any],
    order_generator: Any,
    scaler: Any | None = None,
) -> Path:
    """Write a model checkpoint plus a safe optimizer/RNG resume shard."""
    if step < 0:
        raise ValidationError("checkpoint step must be non-negative")
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch and numpy are required to save a training checkpoint") from exc

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step-{step:08d}"
    weights_path = directory / f"{stem}.safetensors"
    state_path = directory / f"{stem}.state.safetensors"
    metadata_path = directory / f"{stem}.json"
    if any(path.exists() for path in (weights_path, state_path, metadata_path)):
        raise ValidationError(f"Refusing to overwrite existing checkpoint at step {step}: {directory}")

    model_tensors = {f"model.{key}": value for key, value in model.state_dict().items()}
    id_to_name = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_state = optimizer.state_dict()
    serialized_state: dict[str, Any] = {}
    parameter_groups: list[dict[str, Any]] = []
    tensor_state: dict[str, Any] = {}

    for group_index, (actual_group, serialized_group) in enumerate(zip(optimizer.param_groups, optimizer_state["param_groups"])):
        names = []
        for parameter in actual_group["params"]:
            name = id_to_name.get(id(parameter))
            if name is None:
                raise ValidationError("Optimizer contains a parameter not present in the model")
            names.append(name)
        serialized_ids = serialized_group["params"]
        if len(names) != len(serialized_ids):
            raise ValidationError("Optimizer parameter order could not be mapped to model names")
        options = {key: value for key, value in serialized_group.items() if key != "params"}
        parameter_groups.append({"names": names, "options": _json_value(options)})
        for param_id, name in zip(serialized_ids, names):
            state_values = optimizer_state["state"].get(param_id, {})
            encoded: dict[str, Any] = {}
            for state_key, value in state_values.items():
                if torch.is_tensor(value):
                    tensor_key = f"optimizer.{group_index}.{name}.{state_key}"
                    tensor_state[tensor_key] = value
                    encoded[state_key] = {"tensor": tensor_key}
                else:
                    encoded[state_key] = {"value": _json_value(value)}
            if encoded:
                serialized_state[name] = encoded

    cpu_rng = torch.get_rng_state()
    tensor_state["rng.torch_cpu"] = cpu_rng
    model_device = next(model.parameters()).device
    if model_device.type == "cuda" and torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            tensor_state[f"rng.torch_cuda.{index}"] = state
    tensor_state["rng.batch_order"] = order_generator.get_state()
    numpy_state = np.random.get_state()
    tensor_state["rng.numpy_keys"] = torch.from_numpy(numpy_state[1].astype(np.int64, copy=True))
    python_state = random.getstate()

    reserved = {
        "artifact_type", "architecture_id", "model_category", "model_version", "status",
        "checkpoint_file", "checkpoint_sha256", "training_state_file", "training_state_sha256",
        "resume_supported", "step", "model_config", "optimizer", "random_state", "progress",
        "weight_provenance", "scaler_state", "checkpoint_created_at",
    }
    collision = reserved.intersection(run_metadata)
    if collision:
        raise ValidationError(f"run_metadata contains reserved checkpoint keys: {sorted(collision)}")
    _safe_tensor_writer(model_tensors, weights_path)
    _safe_tensor_writer(tensor_state, state_path)
    metadata = {
        **run_metadata,
        "artifact_type": ARTIFACT_TYPE,
        "architecture_id": ARCHITECTURE_ID,
        "model_category": "original",
        "model_version": f"experimental-{run_metadata['run_id']}-step-{step:08d}",
        "status": "experimental",
        "checkpoint_created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_file": weights_path.name,
        "checkpoint_sha256": sha256_file(weights_path),
        "training_state_file": state_path.name,
        "training_state_sha256": sha256_file(state_path),
        "resume_supported": True,
        "step": step,
        "model_config": model_config,
        "optimizer": {
            "type": type(optimizer).__name__,
            "parameter_groups": parameter_groups,
            "state": serialized_state,
        },
        "scaler_state": _json_value(scaler.state_dict()) if scaler is not None else {},
        "random_state": {
            "python": _json_value(python_state),
            "numpy": {
                "bit_generator": numpy_state[0],
                "keys_tensor": "rng.numpy_keys",
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cuda_count": len([key for key in tensor_state if key.startswith("rng.torch_cuda.")]),
        },
        "progress": progress,
        "weight_provenance": {
            "initialization": "random_initialization_in_oasis_training_code",
            "pretrained_components": [],
            "external_weights_used": False,
        },
    }
    write_json_atomic(metadata_path, metadata)
    return metadata_path


def restore_checkpoint(
    checkpoint: str | Path,
    *,
    model: Any,
    optimizer: Any,
    order_generator: Any,
    scaler: Any | None = None,
    map_location: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore model, optimizer, and recorded RNG state using safe tensor files."""
    metadata = verify_checkpoint(checkpoint, require_resume_state=True)
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("PyTorch, numpy, and safetensors are required to resume training") from exc

    metadata_path = checkpoint_metadata_path(checkpoint).resolve()
    weights_path = metadata_path.parent / metadata["checkpoint_file"]
    state_path = metadata_path.parent / metadata["training_state_file"]
    loaded_weights = load_file(str(weights_path), device=map_location)
    model_state = {key.removeprefix("model."): value for key, value in loaded_weights.items() if key.startswith("model.")}
    model.load_state_dict(model_state, strict=True)

    state_tensors = load_file(str(state_path), device="cpu")
    current_optimizer = optimizer.state_dict()
    name_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
    saved_groups = metadata["optimizer"]["parameter_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValidationError("Optimizer parameter-group count differs from the saved checkpoint")
    rebuilt_groups = []
    rebuilt_state: dict[int, dict[str, Any]] = {}
    for group_index, (actual_group, saved_group, current_group) in enumerate(
        zip(optimizer.param_groups, saved_groups, current_optimizer["param_groups"])
    ):
        names = [name_by_parameter.get(id(parameter)) for parameter in actual_group["params"]]
        if names != saved_group["names"]:
            raise ValidationError("Optimizer parameter names/order differ from the saved checkpoint")
        rebuilt_group = dict(current_group)
        rebuilt_group.update(saved_group["options"])
        rebuilt_groups.append(rebuilt_group)
        for param_id, name in zip(current_group["params"], names):
            encoded_values = metadata["optimizer"]["state"].get(name, {})
            values: dict[str, Any] = {}
            for state_key, encoded in encoded_values.items():
                if "tensor" in encoded:
                    tensor_key = encoded["tensor"]
                    if tensor_key not in state_tensors:
                        raise ValidationError(f"Missing optimizer tensor {tensor_key!r} in resume shard")
                    values[state_key] = state_tensors[tensor_key]
                else:
                    values[state_key] = encoded.get("value")
            if values:
                rebuilt_state[param_id] = values
    optimizer.load_state_dict({"state": rebuilt_state, "param_groups": rebuilt_groups})
    if scaler is not None:
        scaler_state = metadata.get("scaler_state", {})
        if scaler_state:
            scaler.load_state_dict(scaler_state)

    torch.set_rng_state(state_tensors["rng.torch_cpu"].to(dtype=torch.uint8))
    cuda_count = metadata["random_state"].get("torch_cuda_count", 0)
    if cuda_count and not torch.cuda.is_available():
        raise ValidationError("Checkpoint contains CUDA RNG state but CUDA is not available for resume")
    if cuda_count and torch.cuda.is_available():
        if cuda_count != torch.cuda.device_count():
            raise ValidationError(
                f"Checkpoint recorded {cuda_count} CUDA devices but {torch.cuda.device_count()} are available; "
                "refusing a non-identical RNG resume"
            )
        cuda_states = [state_tensors[f"rng.torch_cuda.{index}"].to(dtype=torch.uint8) for index in range(cuda_count)]
        torch.cuda.set_rng_state_all(cuda_states)
    order_generator.set_state(state_tensors["rng.batch_order"].to(dtype=torch.uint8))

    numpy_meta = metadata["random_state"]["numpy"]
    np.random.set_state(
        (
            numpy_meta["bit_generator"],
            state_tensors[numpy_meta["keys_tensor"]].cpu().numpy().astype("uint32", copy=False),
            int(numpy_meta["position"]),
            int(numpy_meta["has_gauss"]),
            float(numpy_meta["cached_gaussian"]),
        )
    )
    random.setstate(_tuple_tree(metadata["random_state"]["python"]))
    return metadata, metadata["progress"]
