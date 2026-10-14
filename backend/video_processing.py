"""
Frame extraction (SRS 7.1: "Upload & frame extraction").

Pulls frames from a video at a configurable interval (seconds) using
OpenCV. Kept deliberately synchronous — SRS explicitly says no queue /
async needed at this scale, add it later only if videos get too big.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2

from .models import Frame
from . import storage


def extract_frames(job_id: str, video_path: Path, interval_sec: float = 1.0) -> List[Frame]:
    """
    Extract frames from video_path at `interval_sec` intervals, save them as
    jpgs under jobs/<job_id>/frames/, and return Frame objects (no
    annotations yet — that's the detection pass).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(int(round(fps * interval_sec)), 1)

    out_dir = storage.frames_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[Frame] = []
    raw_index = 0
    saved_index = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if raw_index % frame_step == 0:
            timestamp = raw_index / fps
            filename = f"frame_{saved_index:06d}.jpg"
            out_path = out_dir / filename
            cv2.imwrite(str(out_path), img)
            frames.append(
                Frame(
                    frame_number=saved_index,
                    timestamp=round(timestamp, 3),
                    image_path=f"frames/{filename}",
                )
            )
            saved_index += 1
        raw_index += 1

    cap.release()
    return frames
