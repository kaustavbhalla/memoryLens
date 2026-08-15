"""UnknownPersonBuffer — accumulates face crops across frames before enrollment."""

from __future__ import annotations

import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class UnknownPersonBuffer:
    """Buffers face crops for an unknown person across multiple frames."""
    face_id: str
    face_crops: list[str] = field(default_factory=list)  # temp file paths
    best_face_quality: float = 0.0
    _counter: int = 0

    def add_face(self, crop_b64: str) -> dict:
        """Store a face crop. Returns current buffer status."""
        self._counter += 1
        # Write to temp file
        import base64
        tmp = Path(tempfile.mktemp(suffix=f"_face_{self._counter}.jpg"))
        data = base64.b64decode(crop_b64)
        tmp.write_bytes(data)
        self.face_crops.append(str(tmp))

        # Simple quality heuristic: file size as proxy for detail
        quality = min(len(data) / 10000.0, 1.0)
        if quality > self.best_face_quality:
            self.best_face_quality = quality

        return self.status()

    def status(self) -> dict:
        return {
            "crop_count": len(self.face_crops),
            "best_face_quality": self.best_face_quality,
            "ready": len(self.face_crops) >= 2 and self.best_face_quality >= 0.5,
        }

    def best_crop_path(self) -> str | None:
        if not self.face_crops:
            return None
        return max(self.face_crops, key=lambda p: os.path.getsize(p))

    def cleanup(self):
        for p in self.face_crops:
            try:
                os.unlink(p)
            except OSError:
                pass
        self.face_crops.clear()


# Module-level registry
_buffers: dict[str, UnknownPersonBuffer] = {}


def get_or_create_buffer(face_id: str) -> UnknownPersonBuffer:
    if face_id not in _buffers:
        _buffers[face_id] = UnknownPersonBuffer(face_id=face_id)
    return _buffers[face_id]


def cleanup_buffer(face_id: str):
    buf = _buffers.pop(face_id, None)
    if buf:
        buf.cleanup()
