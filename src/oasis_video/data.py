"""Prepared clip loader for the research training and evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import ValidationError
from .manifests import validate_manifest
from .provenance import load_registry


class VideoClipDataset:
    """Loads fixed PNG sequences lazily, returning normalized [C,T,H,W] tensors."""

    def __init__(
        self,
        manifest_path: str | Path,
        registry_path: str | Path,
        *,
        split: str,
        frames: int,
        width: int,
        height: int,
        max_text_bytes: int,
        verify_frame_hashes: bool = False,
    ) -> None:
        try:
            import numpy  # noqa: F401
            import torch  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Training data loading requires numpy, Pillow, and PyTorch") from exc

        self.manifest_path = Path(manifest_path).resolve()
        registry = load_registry(registry_path)
        self.rows = validate_manifest(
            self.manifest_path,
            registry,
            require_prepared=True,
            for_training=True,
            split=split,
            verify_frame_hashes=verify_frame_hashes,
        )
        self.frames = frames
        self.width = width
        self.height = height
        self.max_text_bytes = max_text_bytes
        for row in self.rows:
            if row["frame_count"] != frames or row["width"] != width or row["height"] != height:
                raise ValidationError(
                    f"Prepared clip {row['item_id']!r} has {row['frame_count']}x{row['height']}x{row['width']}; "
                    f"model expects {frames}x{height}x{width}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        import numpy as np
        import torch
        from PIL import Image

        from .model import encode_caption_ids

        row = self.rows[index]
        data_root = (self.manifest_path.parent / row.get("data_root", ".")).resolve()
        frames = []
        for record in row["frame_paths"]:
            relative = record if isinstance(record, str) else record["path"]
            frame_path = data_root / Path(*relative.replace("\\", "/").split("/"))
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB")
                if rgb.size != (self.width, self.height):
                    raise ValidationError(
                        f"Frame {frame_path} is {rgb.width}x{rgb.height}; expected {self.width}x{self.height}"
                    )
                array = np.array(rgb, dtype=np.uint8, copy=True)
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        video = torch.stack(frames, dim=1).to(dtype=torch.float32).div_(127.5).sub_(1.0)
        token_ids = torch.tensor(encode_caption_ids(row["caption"], self.max_text_bytes), dtype=torch.long)
        return video, token_ids, row["caption"]
