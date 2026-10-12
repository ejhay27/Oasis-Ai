"""JSONL source/prepared manifests and their integrity checks."""

from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .io_utils import ValidationError, read_jsonl, sha256_file
from .provenance import assert_training_eligible

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = {"train", "validation", "test"}


def validate_manifest(
    path: str | Path,
    registry: dict[str, dict[str, Any]],
    *,
    require_prepared: bool = False,
    for_training: bool = False,
    split: str | None = None,
    verify_frame_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Validate JSONL rows and return them.

    Source manifests contain one source video per row. Prepared manifests contain
    fixed-size PNG frame sequences. Training always requires approved registry
    entries, reviewed captions, prepared frames, and a split assignment.
    """
    source = Path(path).resolve()
    rows = read_jsonl(source)
    if not rows:
        raise ValidationError(f"{source}: manifest contains no rows")
    seen: set[str] = set()

    for line_number, row in enumerate(rows, start=1):
        prefix = f"{source}:{line_number}"
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValidationError(f"{prefix}: item_id must be a non-empty string")
        if item_id in seen:
            raise ValidationError(f"{prefix}: duplicate item_id {item_id!r}")
        seen.add(item_id)

        dataset_id = row.get("dataset_id")
        dataset_version = row.get("dataset_version")
        if not isinstance(dataset_id, str) or not isinstance(dataset_version, str):
            raise ValidationError(f"{prefix}: dataset_id and dataset_version are required strings")
        assert_training_eligible(dataset_id, dataset_version, registry)

        source_path = row.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValidationError(f"{prefix}: source_path must be a non-empty string")
        source_hash = row.get("source_sha256")
        if source_hash is not None and (not isinstance(source_hash, str) or not _SHA256_RE.fullmatch(source_hash)):
            raise ValidationError(f"{prefix}: source_sha256 must be a lowercase SHA-256 digest when supplied")

        caption = row.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValidationError(f"{prefix}: caption must be non-empty; do not infer accuracy from an online label")
        caption_reviewed = row.get("caption_reviewed", False)
        if not isinstance(caption_reviewed, bool):
            raise ValidationError(f"{prefix}: caption_reviewed must be a boolean")
        if for_training and not caption_reviewed:
            raise ValidationError(f"{prefix}: caption has not been manually reviewed")

        if require_prepared or for_training:
            _validate_prepared_row(
                row,
                prefix,
                source.parent,
                verify_frame_hashes=verify_frame_hashes,
                require_split=for_training,
                require_frame_hashes=for_training,
            )
        elif "frame_paths" in row:
            _validate_prepared_row(row, prefix, source.parent, verify_frame_hashes=verify_frame_hashes)

        if row.get("split") is not None and row["split"] not in _SPLITS:
            raise ValidationError(f"{prefix}: split must be one of {sorted(_SPLITS)}")

    if for_training:
        split_by_group: dict[str, str] = {}
        split_by_source_hash: dict[str, str] = {}
        for row in rows:
            assigned_split = row["split"]
            group_id = row["split_group_id"]
            prior_group_split = split_by_group.setdefault(group_id, assigned_split)
            prior_source_split = split_by_source_hash.setdefault(row["source_sha256"], assigned_split)
            if prior_group_split != assigned_split:
                raise ValidationError(f"{source}: split group {group_id!r} leaks across splits")
            if prior_source_split != assigned_split:
                raise ValidationError(f"{source}: source SHA-256 {row['source_sha256']} leaks across splits")

    selected = [row for row in rows if split is None or row.get("split") == split]
    if not selected:
        suffix = f" for split {split!r}" if split else ""
        raise ValidationError(f"{source}: no rows found{suffix}")
    return selected


def _validate_prepared_row(
    row: dict[str, Any],
    prefix: str,
    manifest_dir: Path,
    *,
    verify_frame_hashes: bool,
    require_split: bool = False,
    require_frame_hashes: bool = False,
) -> None:
    source_id = row.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValidationError(f"{prefix}: prepared row requires a non-empty source_id")
    source_hash = row.get("source_sha256")
    if not isinstance(source_hash, str) or not _SHA256_RE.fullmatch(source_hash):
        raise ValidationError(f"{prefix}: prepared row requires a lowercase source_sha256")
    frame_paths = row.get("frame_paths")
    frame_count = row.get("frame_count")
    if not isinstance(frame_paths, list) or not frame_paths:
        raise ValidationError(f"{prefix}: frame_paths must be a non-empty list")
    if not isinstance(frame_count, int) or frame_count < 1 or frame_count != len(frame_paths):
        raise ValidationError(f"{prefix}: frame_count must match the number of frame_paths")

    width, height = row.get("width"), row.get("height")
    fps = row.get("fps")
    if not isinstance(width, int) or width < 1 or not isinstance(height, int) or height < 1:
        raise ValidationError(f"{prefix}: width and height must be positive integers")
    if not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValidationError(f"{prefix}: fps must be a positive finite number")

    data_root = row.get("data_root", ".")
    if not isinstance(data_root, str) or Path(data_root).is_absolute():
        raise ValidationError(f"{prefix}: data_root must be a relative path")
    root_rel = PurePosixPath(data_root.replace("\\", "/"))
    if ".." in root_rel.parts:
        raise ValidationError(f"{prefix}: data_root must not contain '..'")
    root = (manifest_dir / Path(*root_rel.parts)).resolve()
    try:
        root.relative_to(manifest_dir.resolve())
    except ValueError as exc:
        raise ValidationError(f"{prefix}: data_root resolves outside the manifest directory") from exc
    # Keep generated artifacts contained by the manifest directory and declared data root.
    fingerprints = row.get("visual_fingerprints", [])
    if not isinstance(fingerprints, list) or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value) for value in fingerprints
    ):
        raise ValidationError(f"{prefix}: visual_fingerprints must be a list of 16-character hex hashes")

    for frame_index, frame in enumerate(frame_paths):
        if isinstance(frame, str):
            frame_rel, expected_hash = frame, None
        elif isinstance(frame, dict):
            frame_rel, expected_hash = frame.get("path"), frame.get("sha256")
            if expected_hash is not None and (not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash)):
                raise ValidationError(f"{prefix}: frame_paths[{frame_index}].sha256 is invalid")
        else:
            raise ValidationError(f"{prefix}: frame_paths[{frame_index}] must be a path or object")
        if require_frame_hashes and not expected_hash:
            raise ValidationError(f"{prefix}: training requires a SHA-256 for every processed frame")
        if not isinstance(frame_rel, str) or not frame_rel:
            raise ValidationError(f"{prefix}: frame_paths[{frame_index}].path must be a non-empty string")
        pure = PurePosixPath(frame_rel.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts:
            raise ValidationError(f"{prefix}: frame path must be relative and must not contain '..': {frame_rel!r}")
        resolved = (root / Path(*pure.parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValidationError(f"{prefix}: frame path escapes data_root: {frame_rel!r}") from exc
        if not resolved.is_file():
            raise ValidationError(f"{prefix}: frame file does not exist: {resolved}")
        if verify_frame_hashes and expected_hash and sha256_file(resolved) != expected_hash:
            raise ValidationError(f"{prefix}: frame checksum mismatch: {resolved}")

    if require_split:
        split = row.get("split")
        if split not in _SPLITS:
            raise ValidationError(f"{prefix}: training row requires split in {sorted(_SPLITS)}")
        group_id = row.get("split_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValidationError(f"{prefix}: training row requires split_group_id")
