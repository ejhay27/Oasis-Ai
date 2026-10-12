from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oasis_video.catalog import validate_record
from oasis_video.checkpoint import ARCHITECTURE_ID, ARTIFACT_TYPE, verify_checkpoint
from oasis_video.config import load_config
from oasis_video.io_utils import ValidationError, sha256_file, write_json_atomic, write_jsonl_atomic
from oasis_video.manifests import validate_manifest
from oasis_video.provenance import assert_training_eligible, load_registry
from oasis_video.splits import make_splits


def approved_dataset(dataset_id: str = "licensed-test", version: str = "1.0") -> dict:
    return {
        "dataset_id": dataset_id,
        "name": "Fixture dataset",
        "source_url": "https://example.invalid/dataset",
        "version": version,
        "acquired_at": "2026-01-01",
        "rights_holder": "Fixture owner",
        "license": {"name": "Fixture training license", "url": "https://example.invalid/license", "text_sha256": None},
        "permissions": {"training": "allowed", "commercial_use": "unknown", "redistribution": "prohibited"},
        "attribution_required": True,
        "attribution_text": "Fixture attribution",
        "privacy_restrictions": [],
        "known_limitations": ["Synthetic test metadata only"],
        "status": "approved",
        "review": {
            "reviewed_by": "test reviewer",
            "reviewed_at": "2026-01-02",
            "evidence_uri": "file:fixture-review.txt",
            "notes": "Unit-test fixture; not a real dataset approval",
        },
        "integrity": {"source_manifest_sha256": None, "source_file_count": None},
    }


def prepared_row(item_id: str, source_hash: str, fingerprint: str, frame_path: str) -> dict:
    return {
        "item_id": item_id,
        "source_id": item_id.split("-clip")[0],
        "dataset_id": "licensed-test",
        "dataset_version": "1.0",
        "source_path": f"videos/{item_id}.mp4",
        "source_sha256": source_hash,
        "caption": "A reviewed fixture caption.",
        "caption_reviewed": True,
        "frame_count": 1,
        "width": 16,
        "height": 16,
        "fps": 4.0,
        "duration_seconds": 0.75,
        "start_seconds": 0.0,
        "data_root": ".",
        "frame_paths": [{"path": frame_path, "sha256": "0" * 64}],
        "visual_fingerprints": [fingerprint, fingerprint, fingerprint],
    }


class ProvenanceTests(unittest.TestCase):
    def test_empty_registry_is_valid_and_no_dataset_is_approved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "registry.json"
            write_json_atomic(path, {"schema_version": 1, "datasets": []})
            self.assertEqual(load_registry(path), {})
            with self.assertRaises(ValidationError):
                assert_training_eligible("missing", "1", {}, require_commercial_use=False)

    def test_training_requires_explicit_approval_and_permission(self) -> None:
        approved = approved_dataset()
        self.assertIs(assert_training_eligible("licensed-test", "1.0", {"licensed-test": approved}), approved)
        blocked = dict(approved)
        blocked["status"] = "pending"
        with self.assertRaisesRegex(ValidationError, "not approved"):
            assert_training_eligible("licensed-test", "1.0", {"licensed-test": blocked})
        with self.assertRaisesRegex(ValidationError, "version mismatch"):
            assert_training_eligible("licensed-test", "2.0", {"licensed-test": approved})
        with self.assertRaisesRegex(ValidationError, "commercial use"):
            assert_training_eligible("licensed-test", "1.0", {"licensed-test": approved}, require_commercial_use=True)


class ConfigTests(unittest.TestCase):
    def test_example_config_validates_without_claiming_data_exists(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "tiny-video-diffusion.example.toml")
        self.assertEqual(config.model.frames, 8)
        self.assertEqual(config.training.device, "auto")
        self.assertFalse(config.manifest_path.exists())


class ManifestTests(unittest.TestCase):
    def test_training_manifest_checks_caption_review_and_frame_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / "frame.png"
            frame.write_bytes(b"fixture-not-a-real-png")
            registry_path = root / "registry.json"
            write_json_atomic(registry_path, {"schema_version": 1, "datasets": [approved_dataset()]})
            registry = load_registry(registry_path)
            row = prepared_row("clip-1", "a" * 64, "0" * 16, "frame.png")
            row["frame_paths"][0]["sha256"] = sha256_file(frame)
            row["split"] = "train"
            row["split_group_id"] = "sha256:" + "a" * 64
            manifest = root / "prepared.jsonl"
            write_jsonl_atomic(manifest, [row])
            validated = validate_manifest(manifest, registry, require_prepared=True, for_training=True)
            self.assertEqual(validated[0]["item_id"], "clip-1")

            row["caption_reviewed"] = False
            write_jsonl_atomic(manifest, [row])
            with self.assertRaisesRegex(ValidationError, "not been manually reviewed"):
                validate_manifest(manifest, registry, require_prepared=True, for_training=True)

    def test_prepared_paths_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry_path = root / "registry.json"
            write_json_atomic(registry_path, {"schema_version": 1, "datasets": [approved_dataset()]})
            registry = load_registry(registry_path)
            row = prepared_row("clip-1", "a" * 64, "0" * 16, "../outside.png")
            manifest = root / "prepared.jsonl"
            write_jsonl_atomic(manifest, [row])
            with self.assertRaisesRegex(ValidationError, "must not contain"):
                validate_manifest(manifest, registry, require_prepared=True)


class SplitTests(unittest.TestCase):
    def test_splits_keep_source_and_near_duplicates_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry_path = root / "registry.json"
            write_json_atomic(registry_path, {"schema_version": 1, "datasets": [approved_dataset()]})
            rows = []
            for index, (source_hash, fingerprint) in enumerate(
                [("a" * 64, "0" * 16), ("a" * 64, "0" * 16), ("b" * 64, "0" * 16), ("c" * 64, "f" * 16)]
            ):
                frame = root / f"frame-{index}.png"
                frame.write_bytes(f"fixture-frame-{index}".encode())
                row = prepared_row(f"source-{index}-clip0", source_hash, fingerprint, frame.name)
                row["frame_paths"][0]["sha256"] = sha256_file(frame)
                rows.append(row)
            source_manifest = root / "prepared.jsonl"
            output_manifest = root / "splits.jsonl"
            write_jsonl_atomic(source_manifest, rows)
            _, counts = make_splits(
                source_manifest,
                registry_path,
                output_manifest,
                seed=17,
                train_ratio=0.6,
                validation_ratio=0.2,
                test_ratio=0.2,
                near_duplicate_threshold=4,
            )
            output_rows = [json.loads(line) for line in output_manifest.read_text().splitlines()]
            by_id = {row["item_id"]: row for row in output_rows}
            self.assertEqual(by_id["source-0-clip0"]["split"], by_id["source-1-clip0"]["split"])
            self.assertEqual(by_id["source-0-clip0"]["split"], by_id["source-2-clip0"]["split"])
            self.assertEqual(sum(counts.values()), len(rows))
            metadata = json.loads(Path(str(output_manifest) + ".meta.json").read_text())
            self.assertEqual(metadata["input_manifest_sha256"], sha256_file(source_manifest))
            self.assertEqual(metadata["grouping"]["near_duplicate_pairs_linked"], 2)


class CheckpointVerificationTests(unittest.TestCase):
    def test_hash_verified_sidecar_does_not_require_executable_model_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weight_file = root / "step-00000001.safetensors"
            weight_file.write_bytes(b"test fixture bytes; not an actual safetensors model")
            metadata = {
                "artifact_type": ARTIFACT_TYPE,
                "architecture_id": ARCHITECTURE_ID,
                "checkpoint_file": weight_file.name,
                "checkpoint_sha256": sha256_file(weight_file),
                "model_config": {"frames": 8},
                "step": 1,
            }
            sidecar = root / "step-00000001.json"
            write_json_atomic(sidecar, metadata)
            checked = verify_checkpoint(weight_file)
            self.assertEqual(checked["step"], 1)
            weight_file.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValidationError, "SHA-256 mismatch"):
                verify_checkpoint(sidecar)


class CatalogTests(unittest.TestCase):
    def test_external_and_adapted_fields_are_not_interchangeable(self) -> None:
        external = {
            "model_id": "vendor-model",
            "version": "1",
            "category": "external",
            "status": "experimental",
            "creator": "Vendor",
            "license": "Vendor license",
            "distribution_status": "restricted",
            "known_restrictions": ["No redistribution"],
            "upstream_creator": "Vendor",
            "upstream_source": "https://example.invalid/model",
            "upstream_license": "Vendor license",
            "weight_origin": "Vendor-hosted release",
        }
        validate_record(external)
        external["category"] = "adapted"
        with self.assertRaisesRegex(ValidationError, "adapted models require"):
            validate_record(external)


if __name__ == "__main__":
    unittest.main()
