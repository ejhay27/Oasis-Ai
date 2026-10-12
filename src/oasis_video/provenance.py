"""Dataset-rights registry validation and training eligibility checks.

A registry entry is evidence bookkeeping, not legal advice or an independent
legal opinion.  Training is blocked unless the entry has an explicit approved
status and an explicit training permission.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from .io_utils import ValidationError, read_json

SCHEMA_VERSION = 1
STATUSES = {"pending", "approved", "rejected", "takedown"}
PERMISSIONS = {"allowed", "prohibited", "unknown", "restricted"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,99}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"{path}: expected registry schema_version={SCHEMA_VERSION}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValidationError(f"{path}: 'datasets' must be a list")

    result: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for index, entry in enumerate(datasets):
        prefix = f"{path}: datasets[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        dataset_id = entry.get("dataset_id")
        if not isinstance(dataset_id, str) or not _ID_RE.fullmatch(dataset_id):
            errors.append(f"{prefix}.dataset_id must be a lowercase slug (2-100 characters)")
            continue
        if dataset_id in result:
            errors.append(f"{prefix}: duplicate dataset_id {dataset_id!r}")
            continue
        errors.extend(_validate_entry(entry, prefix))
        result[dataset_id] = entry

    if errors:
        raise ValidationError("Dataset registry validation failed:\n- " + "\n- ".join(errors))
    return result


def _validate_entry(entry: dict[str, Any], prefix: str) -> list[str]:
    errors: list[str] = []
    for key in ("name", "source_url", "version", "rights_holder"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            errors.append(f"{prefix}.{key} must be a non-empty string")

    acquired_at = entry.get("acquired_at")
    try:
        date.fromisoformat(acquired_at)
    except (TypeError, ValueError):
        errors.append(f"{prefix}.acquired_at must be an ISO date (YYYY-MM-DD)")

    license_info = entry.get("license")
    if not isinstance(license_info, dict):
        errors.append(f"{prefix}.license must be an object with name and url")
    else:
        for key in ("name", "url"):
            if not isinstance(license_info.get(key), str) or not license_info[key].strip():
                errors.append(f"{prefix}.license.{key} must be a non-empty string")
        text_hash = license_info.get("text_sha256")
        if text_hash is not None and (not isinstance(text_hash, str) or not _SHA256_RE.fullmatch(text_hash)):
            errors.append(f"{prefix}.license.text_sha256 must be a lowercase SHA-256 digest or null")

    permissions = entry.get("permissions")
    if not isinstance(permissions, dict):
        errors.append(f"{prefix}.permissions must be an object")
    else:
        for key in ("training", "commercial_use", "redistribution"):
            if permissions.get(key) not in PERMISSIONS:
                errors.append(f"{prefix}.permissions.{key} must be one of {sorted(PERMISSIONS)}")

    status = entry.get("status")
    if status not in STATUSES:
        errors.append(f"{prefix}.status must be one of {sorted(STATUSES)}")

    for key in ("attribution_required",):
        if not isinstance(entry.get(key), bool):
            errors.append(f"{prefix}.{key} must be a boolean")
    for key in ("attribution_text",):
        if not isinstance(entry.get(key), str):
            errors.append(f"{prefix}.{key} must be a string (use an empty string when not applicable)")
    for key in ("privacy_restrictions", "known_limitations"):
        value = entry.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            errors.append(f"{prefix}.{key} must be a list of strings")

    review = entry.get("review")
    if not isinstance(review, dict):
        errors.append(f"{prefix}.review must be an object")
    else:
        for key in ("reviewed_by", "reviewed_at", "evidence_uri", "notes"):
            if not isinstance(review.get(key), str):
                errors.append(f"{prefix}.review.{key} must be a string")
        reviewed_at = review.get("reviewed_at")
        if reviewed_at:
            try:
                date.fromisoformat(reviewed_at)
            except ValueError:
                errors.append(f"{prefix}.review.reviewed_at must be an ISO date (YYYY-MM-DD)")
        if status == "approved":
            for key in ("reviewed_by", "reviewed_at", "evidence_uri"):
                if not isinstance(review.get(key), str) or not review[key].strip():
                    errors.append(f"{prefix}: approved entries require review.{key}")

    integrity = entry.get("integrity", {})
    if not isinstance(integrity, dict):
        errors.append(f"{prefix}.integrity must be an object")
    else:
        manifest_hash = integrity.get("source_manifest_sha256")
        if manifest_hash is not None and (not isinstance(manifest_hash, str) or not _SHA256_RE.fullmatch(manifest_hash)):
            errors.append(f"{prefix}.integrity.source_manifest_sha256 must be a SHA-256 digest or null")
        count = integrity.get("source_file_count")
        if count is not None and (not isinstance(count, int) or count < 0):
            errors.append(f"{prefix}.integrity.source_file_count must be a non-negative integer or null")
    return errors


def assert_training_eligible(
    dataset_id: str,
    dataset_version: str,
    registry: dict[str, dict[str, Any]],
    *,
    require_commercial_use: bool = False,
) -> dict[str, Any]:
    entry = registry.get(dataset_id)
    if entry is None:
        raise ValidationError(f"Dataset {dataset_id!r} is not present in the rights registry")
    if entry.get("version") != dataset_version:
        raise ValidationError(
            f"Dataset {dataset_id!r} version mismatch: manifest has {dataset_version!r}, "
            f"registry has {entry.get('version')!r}"
        )
    if entry.get("status") != "approved":
        raise ValidationError(f"Dataset {dataset_id!r} is not approved (status={entry.get('status')!r})")
    permissions = entry.get("permissions", {})
    if permissions.get("training") != "allowed":
        raise ValidationError(f"Dataset {dataset_id!r} does not have explicit training permission")
    if require_commercial_use and permissions.get("commercial_use") != "allowed":
        raise ValidationError(f"Dataset {dataset_id!r} is not approved for commercial use")
    return entry
