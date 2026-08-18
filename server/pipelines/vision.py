"""Face detection (YOLO) + recognition (DeepFace)."""

import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from server.config import FACE_DETECTION_CONFIDENCE


class VisionPipeline:
    """
    Two-stage face processing:
    1. YOLOv8n-face detects faces (fast, ~5ms CPU)
    2. DeepFace ArcFace extracts 512-dim embeddings (slower, on detected crops)
    """

    def __init__(self):
        self.detector: YOLO | None = None

    async def load(self):
        self.detector = YOLO("yolov8n.pt")

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """Detect faces in a frame. Returns list of {bbox, confidence, crop}."""
        if self.detector is None:
            raise RuntimeError("VisionPipeline not loaded — call load() first")

        results = self.detector(frame, verbose=False)[0]
        faces = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > FACE_DETECTION_CONFIDENCE:
                # Clamp to frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    faces.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "crop": frame[y1:y2, x1:x2].copy(),
                    })
        return faces

    @staticmethod
    def extract_embedding(face_crop: np.ndarray) -> np.ndarray | None:
        """Extract 512-dim ArcFace embedding from a face crop."""
        try:
            result = DeepFace.represent(
                img_path=face_crop,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False,
            )
            return np.array(result[0]["embedding"], dtype=np.float32)
        except Exception:
            return None
