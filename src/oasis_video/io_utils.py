"""Small, dependency-free helpers for stable research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


class ValidationError(ValueError):
    """Raised when a research artifact fails validation."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _write_bytes_atomic(destination, payload.encode("utf-8"))


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    _write_bytes_atomic(destination, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def _write_bytes_atomic(destination: Path, payload: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def read_json(path: str | Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Could not read JSON file {path}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(value, dict):
                    raise ValidationError(f"{source}:{line_number}: each JSONL row must be an object")
                rows.append(value)
    except OSError as exc:
        raise ValidationError(f"Could not read JSONL file {source}: {exc}") from exc
    return rows
