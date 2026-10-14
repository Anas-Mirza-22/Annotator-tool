"""
Detection pass (SRS 7.2): run a pretrained YOLOv8 model over each extracted
frame and store boxes + confidence + label. One detector, one pass — no
multi-model comparison, no tracking (explicit non-goals for v1).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .models import Annotation, Frame
from . import storage

_model_cache = {}


def _get_model(weights: str = "yolo11n.pt"):
    """Lazily load + cache the YOLO model (loading is the slow part)."""
    if weights not in _model_cache:
        from ultralytics import YOLO  # imported lazily so the rest of the
        # app works even before ultralytics/torch are installed
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def run_detection(
    job_id: str,
    frames: List[Frame],
    weights: str = "yolo11n.pt",
    conf_threshold: float = 0.25,
    auto_approve_threshold: Optional[float] = 0.85,
) -> None:
    """
    Run detection on every frame in `frames`, mutating each Frame's
    `annotations` list in place with the model's predictions.

    Boxes are the AI's first pass either way (source="ai"). Confidence
    decides how much of your time they need:
      - confidence >= auto_approve_threshold -> status="approved"
        (the model is confident; skip reviewing these unless you spot a
        problem while scanning past them)
      - confidence <  auto_approve_threshold -> status="pending"
        (the model is unsure; these are the ones that actually need your
        eyes)

    Set auto_approve_threshold=None to disable auto-approval and leave
    every box "pending" (the old behavior).
    """
    model = _get_model(weights)
    job_folder = storage.job_dir(job_id)

    for frame in frames:
        image_path = job_folder / frame.image_path
        results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)
        result = results[0]

        annotations: List[Annotation] = []
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            status = (
                "approved"
                if auto_approve_threshold is not None and conf >= auto_approve_threshold
                else "pending"
            )
            annotations.append(
                Annotation(
                    label=label,
                    bbox=[x1, y1, x2 - x1, y2 - y1],  # store as x, y, w, h
                    confidence=round(conf, 4),
                    source="ai",
                    status=status,
                )
            )
        frame.annotations = annotations