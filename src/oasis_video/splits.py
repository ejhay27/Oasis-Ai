"""Stable group-aware train/validation/test splitting and duplicate screening."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import ValidationError, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from .manifests import validate_manifest
from .provenance import load_registry


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _near_duplicate_pairs(rows: list[dict[str, Any]], threshold: int) -> list[tuple[int, int, list[int]]]:
    if threshold <= 0:
        return []
    if threshold > 8:
        raise ValidationError("near-duplicate Hamming threshold must be between 0 and 8")

    # A 64-bit fingerprint is divided into threshold+1 chunks. Any pair within
    # the configured Hamming radius must share at least one chunk. This avoids
    # a full all-pairs scan while retaining all candidates for that radius.
    chunks = threshold + 1
    boundaries = [round(index * 64 / chunks) for index in range(chunks + 1)]
    buckets: dict[tuple[int, int, str], list[tuple[int, int, str]]] = {}
    candidates: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        fingerprints = row.get("visual_fingerprints", [])
        if not isinstance(fingerprints, list):
            continue
        for frame_position, fingerprint in enumerate(fingerprints):
            if not isinstance(fingerprint, str) or len(fingerprint) != 16:
                continue
            value = int(fingerprint, 16)
            for chunk_index in range(chunks):
                start, end = boundaries[chunk_index], boundaries[chunk_index + 1]
                width = end - start
                mask = (1 << width) - 1
                chunk_value = f"{(value >> start) & mask:0{(width + 3) // 4}x}"
                bucket_key = (frame_position, chunk_index, chunk_value)
                for other_index, other_position, _ in buckets.get(bucket_key, []):
                    if row_index != other_index:
                        candidates.add((other_index, row_index))
                buckets.setdefault(bucket_key, []).append((row_index, frame_position, fingerprint))

    results: list[tuple[int, int, list[int]]] = []
    for left, right in sorted(candidates):
        if rows[left].get("source_sha256") == rows[right].get("source_sha256"):
            continue
        left_hashes = rows[left].get("visual_fingerprints", [])
        right_hashes = rows[right].get("visual_fingerprints", [])
        distances: list[int] = []
        for position in range(min(len(left_hashes), len(right_hashes))):
            distances.append(_hamming(left_hashes[position], right_hashes[position]))
        matched = [distance for distance in distances if distance <= threshold]
        required_matches = 2 if min(len(left_hashes), len(right_hashes)) >= 3 else 1
        if len(matched) >= required_matches:
            results.append((left, right, distances))
    return results


def _split_for_group(group_key: str, seed: int, train: float, validation: float) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()[:8], "big") / 2**64
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def make_splits(
    manifest_path: str | Path,
    registry_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 2026,
    train_ratio: float = 0.9,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    near_duplicate_threshold: int = 4,
) -> tuple[Path, dict[str, int]]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not math.isfinite(value) or value < 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValidationError("train, validation, and test ratios must be non-negative and sum to 1")
    if train_ratio <= 0 or validation_ratio <= 0 or test_ratio <= 0:
        raise ValidationError("all three split ratios must be positive")
    if near_duplicate_threshold < 0 or near_duplicate_threshold > 8:
        raise ValidationError("near-duplicate Hamming threshold must be between 0 and 8")

    source = Path(manifest_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValidationError("Split output must be a new file; do not mutate the prepared manifest")
    if source.parent != destination.parent:
        raise ValidationError("Split output must be next to the input manifest so relative frame paths remain valid")
    meta_destination = Path(str(destination) + ".meta.json")
    if destination.exists() or meta_destination.exists():
        raise ValidationError(f"Refusing to overwrite an existing split manifest or metadata file: {destination}")
    input_manifest_hash = sha256_file(source)
    input_meta_path = Path(str(source) + ".meta.json")
    input_meta_hash = sha256_file(input_meta_path) if input_meta_path.is_file() else None
    registry = load_registry(registry_path)
    rows = validate_manifest(source, registry, require_prepared=True)
    union_find = _UnionFind(len(rows))

    # All clips from the same source video stay together, independent of clips'
    # temporal offsets. Exact source hashes also bind duplicate source files.
    first_by_source: dict[str, int] = {}
    for index, row in enumerate(rows):
        source_hash = row.get("source_sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValidationError(f"{source}: row {row['item_id']!r} lacks a source_sha256")
        prior = first_by_source.setdefault(source_hash, index)
        if prior != index:
            union_find.union(prior, index)

    near_pairs = _near_duplicate_pairs(rows, near_duplicate_threshold)
    for left, right, _ in near_pairs:
        union_find.union(left, right)

    members: dict[int, list[int]] = {}
    for index in range(len(rows)):
        members.setdefault(union_find.find(index), []).append(index)

    counts = {"train": 0, "validation": 0, "test": 0}
    output_rows = [dict(row) for row in rows]
    for component in members.values():
        source_hashes = sorted({rows[index]["source_sha256"] for index in component})
        group_key = source_hashes[0]
        split = _split_for_group(group_key, seed, train_ratio, validation_ratio)
        group_id = f"sha256:{group_key}"
        for index in component:
            output_rows[index]["split"] = split
            output_rows[index]["split_group_id"] = group_id
            counts[split] += 1

    write_jsonl_atomic(destination, output_rows)
    meta = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": source.name,
        "input_manifest_sha256": input_manifest_hash,
        "input_manifest_meta_sha256": input_meta_hash,
        "registry_sha256": sha256_file(registry_path),
        "output_manifest": destination.name,
        "output_manifest_sha256": sha256_file(destination),
        "algorithm": "stable_sha256_group_assignment_v1",
        "grouping": {
            "same_source_sha256": True,
            "visual_fingerprint": "64-bit grayscale average hash; heuristic candidate grouping only",
            "near_duplicate_hamming_threshold": near_duplicate_threshold,
            "near_duplicate_pairs_linked": len(near_pairs),
            "near_duplicate_pairs": [
                {
                    "left_item_id": rows[left]["item_id"],
                    "right_item_id": rows[right]["item_id"],
                    "hamming_distances_by_sampled_frame": distances,
                }
                for left, right, distances in near_pairs
            ],
            "manual_review_required": True,
        },
        "seed": seed,
        "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "counts": counts,
        "groups": len(members),
    }
    write_json_atomic(str(destination) + ".meta.json", meta)
    return destination, counts
