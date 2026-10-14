"""
Export module (SRS 7.4): turn the reviewed annotation state into a format
the client actually asked for. Only annotations with status
"approved" or "edited" are exported — "pending" (never reviewed) and
"rejected" boxes are dropped. This is the human-QC gate the whole tool
exists for.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import Job
from . import storage

INCLUDED_STATUSES = ("approved", "edited")


def _included_annotations(job: Job):
    for frame in job.frames:
        for ann in frame.annotations:
            if ann.status in INCLUDED_STATUSES:
                yield frame, ann


def export_json(job: Job) -> Path:
    """Plain JSON export — closest to the internal data model."""
    out = {
        "job_id": job.job_id,
        "client_name": job.client_name,
        "video_file": job.video_file,
        "frames": [
            {
                "frame_number": frame.frame_number,
                "timestamp": frame.timestamp,
                "image_path": frame.image_path,
                "annotations": [
                    {
                        "label": ann.label,
                        "bbox": ann.bbox,
                        "confidence": ann.confidence,
                        "source": ann.source,
                    }
                    for ann in frame.annotations
                    if ann.status in INCLUDED_STATUSES
                ],
            }
            for frame in job.frames
        ],
    }
    out_path = storage.exports_dir(job.job_id) / "export.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out_path


def export_coco(job: Job) -> Path:
    """COCO-style JSON: images[], annotations[], categories[]."""
    categories: Dict[str, int] = {}
    images = []
    annotations = []
    ann_id = 1

    for frame in job.frames:
        image_id = frame.frame_number
        images.append(
            {
                "id": image_id,
                "file_name": Path(frame.image_path).name,
                "frame_number": frame.frame_number,
                "timestamp": frame.timestamp,
            }
        )
        for ann in frame.annotations:
            if ann.status not in INCLUDED_STATUSES:
                continue
            if ann.label not in categories:
                categories[ann.label] = len(categories) + 1
            x, y, w, h = ann.bbox
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": categories[ann.label],
                    "bbox": [x, y, w, h],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "score": ann.confidence,
                }
            )
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name} for name, cid in categories.items()],
    }
    out_path = storage.exports_dir(job.job_id) / "export_coco.json"
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)
    return out_path


def export_yolo(job: Job) -> Path:
    """
    YOLO format: one .txt per frame with normalized
    `class_id cx cy w h`, plus a classes.txt mapping id -> label.
    Requires the source image dimensions to normalize the boxes.
    """
    import cv2

    out_dir = storage.exports_dir(job.job_id) / "yolo"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: Dict[str, int] = {}
    job_folder = storage.job_dir(job.job_id)

    for frame in job.frames:
        included = [a for a in frame.annotations if a.status in INCLUDED_STATUSES]
        img_path = job_folder / frame.image_path
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        lines: List[str] = []
        for ann in included:
            if ann.label not in labels:
                labels[ann.label] = len(labels)
            x, y, w, h = ann.bbox
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{labels[ann.label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        txt_name = Path(frame.image_path).stem + ".txt"
        with open(out_dir / txt_name, "w") as f:
            f.write("\n".join(lines))

    with open(out_dir / "classes.txt", "w") as f:
        for name, idx in sorted(labels.items(), key=lambda kv: kv[1]):
            f.write(f"{name}\n")

    return out_dir


EXPORTERS = {
    "json": export_json,
    "coco": export_coco,
    "yolo": export_yolo,
}


def export(job: Job, fmt: str) -> Path:
    if fmt not in EXPORTERS:
        raise ValueError(f"Unknown export format '{fmt}'. Choose from: {list(EXPORTERS)}")
    return EXPORTERS[fmt](job)
