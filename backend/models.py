"""
Core data model for Annotator Tool (see SRS section 6).

Deliberately simple dataclasses, JSON-serializable. No ORM, no DB —
everything is persisted as plain JSON files inside each job's folder
(see storage.py). One person, one process, local disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import time
import uuid


VALID_JOB_STATUSES = ("uploaded", "processing", "in_review", "done")
VALID_ANN_SOURCES = ("ai", "human_corrected")
VALID_ANN_STATUSES = ("pending", "approved", "rejected", "edited")


@dataclass
class Annotation:
    label: str
    bbox: List[float]  # [x, y, w, h] in pixel coords, top-left origin
    confidence: float = 1.0
    source: str = "ai"          # ai | human_corrected
    status: str = "pending"     # pending | approved | rejected | edited
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Annotation":
        return Annotation(
            label=d["label"],
            bbox=list(d["bbox"]),
            confidence=d.get("confidence", 1.0),
            source=d.get("source", "ai"),
            status=d.get("status", "pending"),
            id=d.get("id", uuid.uuid4().hex[:8]),
        )


@dataclass
class Frame:
    frame_number: int
    timestamp: float           # seconds into the video
    image_path: str            # relative path within the job folder
    annotations: List[Annotation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "annotations": [a.to_dict() for a in self.annotations],
        }

    @staticmethod
    def from_dict(d: dict) -> "Frame":
        return Frame(
            frame_number=d["frame_number"],
            timestamp=d["timestamp"],
            image_path=d["image_path"],
            annotations=[Annotation.from_dict(a) for a in d.get("annotations", [])],
        )


@dataclass
class Job:
    client_name: str
    video_file: str = ""        # relative path within the job folder; "" for image-only jobs
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "uploaded"    # uploaded | processing | in_review | done
    created_at: float = field(default_factory=time.time)
    frame_interval_sec: float = 1.0
    source_type: str = "video"  # "video" | "images" — how the frames were produced
    frames: List[Frame] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "client_name": self.client_name,
            "video_file": self.video_file,
            "status": self.status,
            "created_at": self.created_at,
            "frame_interval_sec": self.frame_interval_sec,
            "source_type": self.source_type,
            "frames": [f.to_dict() for f in self.frames],
        }

    @staticmethod
    def from_dict(d: dict) -> "Job":
        return Job(
            job_id=d["job_id"],
            client_name=d["client_name"],
            video_file=d.get("video_file", ""),
            status=d.get("status", "uploaded"),
            created_at=d.get("created_at", time.time()),
            frame_interval_sec=d.get("frame_interval_sec", 1.0),
            source_type=d.get("source_type", "video"),
            frames=[Frame.from_dict(f) for f in d.get("frames", [])],
        )