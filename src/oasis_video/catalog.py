"""Versioned catalog that keeps external, adapted, and original models distinct."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint import verify_checkpoint
from .io_utils import ValidationError, read_json, sha256_file, write_json_atomic

CATEGORIES = {"external", "adapted", "original"}
STATUSES = {"experimental", "research_preview", "internal_alpha", "limited_release", "production", "retired"}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_record(record: dict[str, Any], prefix: str = "model") -> None:
    for field in ("model_id", "version", "creator", "license", "distribution_status"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValidationError(f"{prefix}.{field} must be a non-empty string")
    if record.get("category") not in CATEGORIES:
        raise ValidationError(f"{prefix}.category must be one of {sorted(CATEGORIES)}")
    if record.get("status") not in STATUSES:
        raise ValidationError(f"{prefix}.status must be one of {sorted(STATUSES)}")
    if record.get("distribution_status") not in {"not_cleared", "internal_only", "cleared", "restricted", "unknown"}:
        raise ValidationError(f"{prefix}.distribution_status must be not_cleared, internal_only, cleared, restricted, or unknown")
    if record.get("status") in {"limited_release", "production"} and record["distribution_status"] != "cleared":
        raise ValidationError(f"{prefix}: public release/production requires distribution_status='cleared'")
    if not isinstance(record.get("known_restrictions"), list) or any(
        not isinstance(item, str) for item in record["known_restrictions"]
    ):
        raise ValidationError(f"{prefix}.known_restrictions must be a list of strings")

    category = record["category"]
    if category == "external":
        for field in ("upstream_creator", "upstream_source", "upstream_license", "weight_origin"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValidationError(f"{prefix}: external models require {field}")
    elif category == "adapted":
        for field in (
            "base_model_id", "base_model_license", "fine_tuning_method", "training_manifest_sha256",
            "inherited_weights", "new_weights_description", "redistribution_permission",
        ):
            if field not in record:
                raise ValidationError(f"{prefix}: adapted models require {field}")
        if not isinstance(record["inherited_weights"], list) or not isinstance(record["fine_tuning_method"], str):
            raise ValidationError(f"{prefix}: adapted-model provenance fields have invalid types")
        if not isinstance(record["training_manifest_sha256"], str) or not _HASH_RE.fullmatch(record["training_manifest_sha256"]):
            raise ValidationError(f"{prefix}.training_manifest_sha256 must be a lowercase SHA-256 digest")
    else:
        for field in (
            "architecture_id", "training_run_id", "training_report", "training_report_sha256", "dataset_manifest_sha256",
            "dataset_registry_sha256", "dataset_versions", "training_config_sha256", "pretrained_components", "weight_initialization",
            "checkpoint_path", "checkpoint_sha256", "checkpoint_metadata_path", "checkpoint_metadata_sha256", "training_step",
        ):
            if field not in record:
                raise ValidationError(f"{prefix}: original models require {field}")
        if not isinstance(record["pretrained_components"], list):
            raise ValidationError(f"{prefix}.pretrained_components must be a list")
        if not isinstance(record["dataset_versions"], list) or any(
            not isinstance(item, dict) or not isinstance(item.get("dataset_id"), str) or not isinstance(item.get("version"), str)
            for item in record["dataset_versions"]
        ):
            raise ValidationError(f"{prefix}.dataset_versions must list dataset_id/version objects")
        if not isinstance(record["weight_initialization"], str):
            raise ValidationError(f"{prefix}.weight_initialization must be a string")
        if isinstance(record["training_step"], bool) or not isinstance(record["training_step"], int) or record["training_step"] < 1:
            raise ValidationError(f"{prefix}.training_step must be a positive integer")
        for field in (
            "dataset_manifest_sha256", "dataset_registry_sha256", "training_config_sha256", "training_report_sha256",
            "checkpoint_sha256", "checkpoint_metadata_sha256",
        ):
            if not isinstance(record.get(field), str) or not _HASH_RE.fullmatch(record[field]):
                raise ValidationError(f"{prefix}.{field} must be a lowercase SHA-256 digest")


def load_catalog(path: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    source = Path(path).resolve()
    catalog = read_json(source)
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1 or not isinstance(catalog.get("models"), list):
        raise ValidationError(f"{source}: expected schema_version=1 and a models list")
    seen: set[str] = set()
    for index, record in enumerate(catalog["models"]):
        if not isinstance(record, dict):
            raise ValidationError(f"{source}: models[{index}] must be an object")
        validate_record(record, f"models[{index}]")
        model_id = record["model_id"]
        if model_id in seen:
            raise ValidationError(f"{source}: duplicate model_id {model_id!r}")
        seen.add(model_id)
        if verify_files and record["category"] == "original":
            checkpoint_path = (source.parent / record["checkpoint_path"]).resolve()
            metadata_path = (source.parent / record["checkpoint_metadata_path"]).resolve()
            if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
                raise ValidationError(f"{source}: checkpoint missing or hash mismatch for {model_id!r}")
            if not metadata_path.is_file() or sha256_file(metadata_path) != record["checkpoint_metadata_sha256"]:
                raise ValidationError(f"{source}: checkpoint sidecar missing or hash mismatch for {model_id!r}")
            training_report = (source.parent / record["training_report"]).resolve()
            if not training_report.is_file() or sha256_file(training_report) != record["training_report_sha256"]:
                raise ValidationError(f"{source}: training report missing or hash mismatch for {model_id!r}")
            metadata = verify_checkpoint(metadata_path)
            if metadata.get("model_version") != model_id:
                raise ValidationError(f"{source}: catalog id does not match checkpoint model_version for {model_id!r}")
            sidecar_checkpoint = (metadata_path.parent / metadata.get("checkpoint_file", "")).resolve()
            if sidecar_checkpoint != checkpoint_path or metadata.get("checkpoint_sha256") != record["checkpoint_sha256"]:
                raise ValidationError(f"{source}: catalog checkpoint path/hash differs from its sidecar for {model_id!r}")
            report = read_json(training_report)
            if (
                report.get("run_id") != record["training_run_id"]
                or report.get("status") not in {"completed", "interrupted"}
                or int(report.get("last_step", -1)) < int(record.get("training_step", 0))
            ):
                raise ValidationError(f"{source}: training report does not verify the registered run/step for {model_id!r}")
            if metadata.get("registry_sha256") != record["dataset_registry_sha256"]:
                raise ValidationError(f"{source}: rights-registry hash mismatch for {model_id!r}")
            if metadata.get("dataset_versions") != record["dataset_versions"] or metadata.get("step") != record["training_step"]:
                raise ValidationError(f"{source}: dataset-version/training-step provenance mismatch for {model_id!r}")
    return catalog


def register_original_checkpoint(checkpoint: str | Path, catalog_path: str | Path) -> dict[str, Any]:
    metadata = verify_checkpoint(checkpoint, require_resume_state=True)
    if metadata.get("model_category") != "original" or metadata.get("status") != "experimental":
        raise ValidationError("Only experimental OASIS-original checkpoints from the fixed training pipeline can be registered here")
    provenance = metadata.get("weight_provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("pretrained_components"), list):
        raise ValidationError("Checkpoint is missing structured learned-weight provenance")
    required = (
        "run_id", "run_directory", "config_sha256", "manifest_sha256", "registry_sha256", "dataset_versions", "step"
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValidationError(f"Checkpoint lacks training provenance fields: {', '.join(missing)}")
    catalog_file = Path(catalog_path).resolve()
    catalog = load_catalog(catalog_file, verify_files=False)
    model_id = metadata["model_version"]
    if any(record.get("model_id") == model_id for record in catalog["models"]):
        raise ValidationError(f"Model version {model_id!r} is already registered")
    checkpoint_metadata = Path(checkpoint).resolve()
    if checkpoint_metadata.suffix.lower() != ".json":
        checkpoint_metadata = checkpoint_metadata.with_suffix(".json")
    checkpoint_file = checkpoint_metadata.parent / metadata["checkpoint_file"]
    training_report_file = Path(metadata["run_directory"]).resolve() / "run.json"
    if not training_report_file.is_file():
        raise ValidationError(f"Checkpoint training report is missing: {training_report_file}")
    training_report = read_json(training_report_file)
    if (
        training_report.get("run_id") != metadata["run_id"]
        or training_report.get("status") not in {"completed", "interrupted"}
        or int(training_report.get("last_step", -1)) < int(metadata["step"])
    ):
        raise ValidationError("Training report status/run/step does not support registering this checkpoint")
    try:
        checkpoint_rel = os.path.relpath(checkpoint_file, catalog_file.parent)
        metadata_rel = os.path.relpath(checkpoint_metadata, catalog_file.parent)
        run_rel = os.path.relpath(metadata["run_directory"], catalog_file.parent)
    except ValueError as exc:
        raise ValidationError(f"Checkpoint cannot be referenced from the catalog: {exc}") from exc

    record = {
        "model_id": model_id,
        "version": model_id.rsplit("-step-", 1)[-1],
        "category": "original",
        "creator": "OASIS project (architecture and training pipeline)",
        "architecture_id": metadata["architecture_id"],
        "training_run_id": metadata["run_id"],
        "training_report": (Path(run_rel) / "run.json").as_posix(),
        "training_report_sha256": sha256_file(training_report_file),
        "dataset_manifest_sha256": metadata["manifest_sha256"],
        "dataset_registry_sha256": metadata["registry_sha256"],
        "dataset_versions": metadata["dataset_versions"],
        "training_config_sha256": metadata["config_sha256"],
        "pretrained_components": metadata["weight_provenance"].get("pretrained_components", []),
        "weight_initialization": metadata["weight_provenance"].get("initialization", "unknown"),
        "checkpoint_path": Path(checkpoint_rel).as_posix(),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_metadata_path": Path(metadata_rel).as_posix(),
        "checkpoint_metadata_sha256": sha256_file(checkpoint_metadata),
        "license": "Undecided; internal research artifact only pending rights and distribution review",
        "distribution_status": "not_cleared",
        "status": "experimental",
        "known_restrictions": [
            "Research checkpoint; not cleared for redistribution or production use",
            "Dataset and weight distribution rights require separate review",
        ],
        "training_step": metadata["step"],
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_record(record)
    catalog["models"].append(record)
    write_json_atomic(catalog_file, catalog)
    return record
