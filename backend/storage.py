"""
Job folder storage.

Layout (per SRS section 7.5 "job folder structure"):

  jobs/<job_id>/
      job.json            <- Job metadata + full frame/annotation state (source of truth)
      video/<original>.mp4
      frames/frame_000001.jpg ...
      exports/export_coco.json | export_yolo/ | export.json

We save job.json after every mutating action (not just at the end), so a
crash mid-review loses at most the in-progress edit, per SRS section 8
("Reliability").
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .models import Job

JOBS_ROOT = Path(os.environ.get("ANNOTATOR_JOBS_ROOT", Path(__file__).resolve().parent.parent / "jobs"))


def job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def frames_dir(job_id: str) -> Path:
    return job_dir(job_id) / "frames"


def video_dir(job_id: str) -> Path:
    return job_dir(job_id) / "video"


def exports_dir(job_id: str) -> Path:
    return job_dir(job_id) / "exports"


def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def init_job_folders(job_id: str) -> None:
    for d in (job_dir(job_id), frames_dir(job_id), video_dir(job_id), exports_dir(job_id)):
        d.mkdir(parents=True, exist_ok=True)


def save_job(job: Job) -> None:
    init_job_folders(job.job_id)
    tmp_path = job_json_path(job.job_id).with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(job.to_dict(), f, indent=2)
    # atomic-ish replace so a crash mid-write can't corrupt job.json
    os.replace(tmp_path, job_json_path(job.job_id))


def load_job(job_id: str) -> Job:
    with open(job_json_path(job_id)) as f:
        return Job.from_dict(json.load(f))


def job_exists(job_id: str) -> bool:
    return job_json_path(job_id).exists()


def list_jobs() -> list[Job]:
    if not JOBS_ROOT.exists():
        return []
    jobs = []
    for entry in sorted(JOBS_ROOT.iterdir()):
        if (entry / "job.json").exists():
            try:
                jobs.append(load_job(entry.name))
            except Exception:
                continue
    return jobs
