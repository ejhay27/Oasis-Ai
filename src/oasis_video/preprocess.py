"""Auditable video probing and fixed-shape frame extraction using FFmpeg."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from .io_utils import ValidationError, sha256_file, write_json_atomic, write_jsonl_atomic
from .manifests import validate_manifest
from .provenance import load_registry


def _require_binary(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise ValidationError(f"Required executable {name!r} was not found on PATH")
    return executable


def _run(command: list[str], *, context: str) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(f"{context} timed out") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ValidationError(f"{context} failed: {details[-1500:]}") from exc
    return result.stdout.strip()


def inspect_video(path: str | Path) -> dict[str, Any]:
    """Probe stream facts. This does not establish rights, quality, or caption accuracy."""
    video = Path(path)
    if not video.is_file():
        raise ValidationError(f"Video file does not exist: {video}")
    ffprobe = _require_binary("ffprobe")
    raw = _run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
            "-of", "json",
            str(video),
        ],
        context=f"ffprobe on {video}",
    )
    try:
        payload = json.loads(raw)
        streams = payload.get("streams", [])
        if not streams:
            raise ValueError("no video stream")
        stream = streams[0]
        duration = stream.get("duration") or payload.get("format", {}).get("duration")
        rate_value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        fps = float(Fraction(rate_value)) if rate_value and rate_value != "0/0" else 0.0
        result = {
            "codec": stream.get("codec_name"),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
            "duration_seconds": float(duration) if duration not in (None, "N/A") else None,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValidationError(f"Could not parse video metadata for {video}: {exc}") from exc
    if not result["duration_seconds"] or not math.isfinite(result["duration_seconds"]):
        raise ValidationError(f"Video duration is unknown or invalid: {video}")
    if result["fps"] <= 0:
        raise ValidationError(f"Video frame rate is unknown or invalid: {video}")
    return result


def average_hash(image_path: str | Path) -> str:
    """Return a compact 64-bit grayscale average hash for screening only."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValidationError("Pillow is required for visual fingerprints; install the 'preprocess' extra") from exc
    with Image.open(image_path) as image:
        pixels = list(image.convert("L").resize((8, 8), Image.Resampling.LANCZOS).getdata())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return f"{value:016x}"


def preprocess_manifest(
    source_manifest: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    *,
    frames: int = 8,
    width: int = 16,
    height: int = 16,
    fps: float = 4.0,
    stride_seconds: float | None = None,
    max_clips_per_source: int | None = None,
) -> Path:
    """Decode reviewed-rights source rows into fixed-size PNG frame clips.

    The extraction is deterministic for a given FFmpeg build and configuration.
    Audio is discarded. Original media is never modified. Captions are copied as
    provided and are not inferred or considered verified by this function.
    """
    if frames < 2 or width < 8 or height < 8 or fps <= 0:
        raise ValidationError("frames must be >=2, width/height >=8, and fps must be positive")
    if stride_seconds is not None and stride_seconds <= 0:
        raise ValidationError("stride_seconds must be positive")
    if max_clips_per_source is not None and max_clips_per_source < 1:
        raise ValidationError("max_clips_per_source must be at least 1")

    ffmpeg = _require_binary("ffmpeg")
    _require_binary("ffprobe")
    registry = load_registry(registry_path)
    sources = validate_manifest(source_manifest, registry, require_prepared=False)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_manifest = root / "prepared.jsonl"
    output_meta = Path(str(output_manifest) + ".meta.json")
    if output_manifest.exists() or output_meta.exists():
        raise ValidationError(f"Refusing to overwrite an existing prepared manifest: {output_manifest}")
    source_manifest_hash = sha256_file(source_manifest)
    registry_hash = sha256_file(registry_path)
    clips_root = root / "clips"
    clips_root.mkdir(exist_ok=True)
    clip_duration = frames / fps
    stride = stride_seconds if stride_seconds is not None else clip_duration
    version = _run([ffmpeg, "-version"], context="reading FFmpeg version").splitlines()[0]
    prepared: list[dict[str, Any]] = []

    for source in sources:
        source_file = Path(source["source_path"])
        if not source_file.is_absolute():
            source_file = Path(source_manifest).resolve().parent / source_file
        source_file = source_file.resolve()
        if not source_file.is_file():
            raise ValidationError(f"Source video does not exist: {source_file}")
        source_hash = sha256_file(source_file)
        claimed_hash = source.get("source_sha256")
        if claimed_hash and claimed_hash != source_hash:
            raise ValidationError(f"Source SHA-256 mismatch for {source_file}")
        probe = inspect_video(source_file)
        if probe["duration_seconds"] < clip_duration:
            raise ValidationError(
                f"Source {source_file} is too short for {frames} frames at {fps:g} fps "
                f"({probe['duration_seconds']:.3f}s < {clip_duration:.3f}s)"
            )

        count = int(math.floor((probe["duration_seconds"] - clip_duration) / stride + 1e-9)) + 1
        if max_clips_per_source is not None:
            count = min(count, max_clips_per_source)
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", source["item_id"]).strip(".-_")[:64] or "item"
        for clip_index in range(count):
            start = clip_index * stride
            clip_name = f"{safe_id}-{source_hash[:10]}-{clip_index:05d}"
            destination = clips_root / clip_name
            if destination.exists():
                raise ValidationError(f"Refusing to overwrite existing processed clip: {destination}")
            temporary = clips_root / f".tmp-{uuid.uuid4().hex}"
            temporary.mkdir()
            try:
                filter_graph = (
                    f"fps={fps:.8g},scale={width}:{height}:force_original_aspect_ratio=increase,"
                    f"crop={width}:{height}"
                )
                _run(
                    [
                        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{start:.6f}", "-i", str(source_file),
                        "-t", f"{clip_duration:.6f}", "-an",
                        "-vf", filter_graph, "-frames:v", str(frames), "-start_number", "0",
                        "-y", str(temporary / "frame_%04d.png"),
                    ],
                    context=f"extracting {clip_name}",
                )
                frame_files = sorted(temporary.glob("frame_*.png"))
                if len(frame_files) != frames:
                    raise ValidationError(
                        f"FFmpeg produced {len(frame_files)} frames for {clip_name}; expected exactly {frames}"
                    )
                fingerprints = [
                    average_hash(frame_files[index])
                    for index in sorted({0, len(frame_files) // 2, len(frame_files) - 1})
                ]
                temporary.replace(destination)
                relative_dir = Path("clips") / clip_name
                prepared.append(
                    {
                        "item_id": clip_name,
                        "source_id": source["item_id"],
                        "dataset_id": source["dataset_id"],
                        "dataset_version": source["dataset_version"],
                        "source_path": source["source_path"],
                        "source_sha256": source_hash,
                        "caption": source["caption"],
                        "caption_reviewed": source.get("caption_reviewed", False),
                        "frame_count": frames,
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "duration_seconds": clip_duration,
                        "start_seconds": start,
                        "data_root": ".",
                        "frame_paths": [
                            {
                                "path": (relative_dir / frame.name).as_posix(),
                                "sha256": sha256_file(destination / frame.name),
                            }
                            for frame in frame_files
                        ],
                        "visual_fingerprints": fingerprints,
                        "preprocessing": {
                            "tool": "ffmpeg",
                            "ffmpeg_version": version,
                            "audio_included": False,
                            "source_width": probe["width"],
                            "source_height": probe["height"],
                            "source_fps": probe["fps"],
                            "target_width": width,
                            "target_height": height,
                            "target_fps": fps,
                            "clip_frames": frames,
                            "filter": filter_graph,
                        },
                    }
                )
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

    write_jsonl_atomic(output_manifest, prepared)
    write_json_atomic(
        output_meta,
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifest": str(Path(source_manifest).resolve()),
            "source_manifest_sha256": source_manifest_hash,
            "registry_sha256": registry_hash,
            "prepared_manifest": output_manifest.name,
            "prepared_manifest_sha256": sha256_file(output_manifest),
            "source_rows": len(sources),
            "prepared_clips": len(prepared),
            "configuration": {
                "frames": frames,
                "width": width,
                "height": height,
                "fps": fps,
                "stride_seconds": stride,
                "max_clips_per_source": max_clips_per_source,
                "audio_included": False,
                "ffmpeg_version": version,
            },
            "notes": [
                "Captions are copied from the source manifest and are not automatically verified.",
                "Visual fingerprints are approximate average-hash values for duplicate screening only.",
            ],
        },
    )
    return output_manifest
