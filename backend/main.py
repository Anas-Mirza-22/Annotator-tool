"""
FastAPI backend for Annotator Tool.

Single-operator, local-only tool (SRS section 2: no accounts, no
multi-tenant, no auth). Endpoints exist so the Streamlit review UI (and
you, via curl/Swagger if needed) can drive the pipeline:

    upload video -> extract frames -> run detection -> review/correct ->
    export

Run with:  uvicorn backend.main:app --reload
Docs at:   http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import storage, video_processing, detection, export as export_mod
from .models import Annotation, Frame, Job

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI(title="Annotator Tool", version="0.1")


# ---------------------------------------------------------------------
# Schemas for request/response bodies
# ---------------------------------------------------------------------

class AnnotationUpdate(BaseModel):
    label: Optional[str] = None
    bbox: Optional[List[float]] = None
    status: Optional[str] = None  # approved | rejected | edited


class AnnotationCreate(BaseModel):
    label: str
    bbox: List[float]
    status: str = "approved"


# ---------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------

@app.get("/jobs")
def api_list_jobs():
    jobs = storage.list_jobs()
    return [
        {
            "job_id": j.job_id,
            "client_name": j.client_name,
            "status": j.status,
            "created_at": j.created_at,
            "num_frames": len(j.frames),
        }
        for j in jobs
    ]


@app.post("/jobs")
def api_create_job(
    client_name: str = Form(...),
    frame_interval_sec: float = Form(1.0),
    video: UploadFile = File(...),
):
    job = Job(client_name=client_name, video_file="", frame_interval_sec=frame_interval_sec)
    storage.init_job_folders(job.job_id)

    video_dest = storage.video_dir(job.job_id) / video.filename
    with open(video_dest, "wb") as f:
        shutil.copyfileobj(video.file, f)
    job.video_file = f"video/{video.filename}"

    job.status = "processing"
    storage.save_job(job)

    try:
        frames = video_processing.extract_frames(job.job_id, video_dest, frame_interval_sec)
    except Exception as e:
        job.status = "uploaded"
        storage.save_job(job)
        raise HTTPException(status_code=400, detail=f"Frame extraction failed: {e}")

    job.frames = frames
    job.status = "uploaded"
    storage.save_job(job)
    return {"job_id": job.job_id, "num_frames": len(frames)}


@app.post("/jobs/images")
def api_create_image_job(
    client_name: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """
    Create a job directly from a batch of still images — no video, no
    frame extraction. Each uploaded image becomes one Frame, in the order
    uploaded. Everything downstream (detection, review, export) works the
    same as a video-derived job.
    """
    bad_files = [
        img.filename for img in images if Path(img.filename).suffix.lower() not in IMAGE_EXTENSIONS
    ]
    if bad_files:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image file(s): {bad_files}. Allowed: {sorted(IMAGE_EXTENSIONS)}",
        )

    job = Job(client_name=client_name, video_file="", source_type="images")
    storage.init_job_folders(job.job_id)

    frames_out = storage.frames_dir(job.job_id)
    frames_out.mkdir(parents=True, exist_ok=True)

    frames: List[Frame] = []
    for i, img in enumerate(images):
        ext = Path(img.filename).suffix.lower()
        filename = f"frame_{i:06d}{ext}"
        dest = frames_out / filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(img.file, f)
        frames.append(Frame(frame_number=i, timestamp=0.0, image_path=f"frames/{filename}"))

    job.frames = frames
    job.status = "uploaded"
    storage.save_job(job)
    return {"job_id": job.job_id, "num_frames": len(frames)}


@app.get("/jobs/{job_id}")
def api_get_job(job_id: str):
    _require_job(job_id)
    job = storage.load_job(job_id)
    return {
        "job_id": job.job_id,
        "client_name": job.client_name,
        "status": job.status,
        "num_frames": len(job.frames),
        "frame_interval_sec": job.frame_interval_sec,
    }


@app.post("/jobs/{job_id}/detect")
def api_run_detection(
    job_id: str,
    weights: str = "yolo11n.pt",
    conf_threshold: float = 0.25,
    auto_approve_threshold: Optional[float] = 0.85,
):
    _require_job(job_id)
    job = storage.load_job(job_id)
    if not job.frames:
        raise HTTPException(status_code=400, detail="No frames extracted for this job yet.")

    job.status = "processing"
    storage.save_job(job)
    try:
        detection.run_detection(
            job_id,
            job.frames,
            weights=weights,
            conf_threshold=conf_threshold,
            auto_approve_threshold=auto_approve_threshold,
        )
    except Exception as e:
        # don't leave the job stuck in "processing" after a failed run —
        # revert so the person can fix the issue and retry cleanly
        job.status = "uploaded"
        storage.save_job(job)
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    job.status = "in_review"
    storage.save_job(job)
    total_boxes = sum(len(f.annotations) for f in job.frames)
    auto_approved = sum(
        1 for f in job.frames for a in f.annotations if a.status == "approved"
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "total_boxes": total_boxes,
        "auto_approved": auto_approved,
        "needs_review": total_boxes - auto_approved,
    }


# ---------------------------------------------------------------------
# Frames + annotations
# ---------------------------------------------------------------------

@app.get("/jobs/{job_id}/frames")
def api_list_frames(job_id: str):
    _require_job(job_id)
    job = storage.load_job(job_id)
    return [
        {
            "frame_number": f.frame_number,
            "timestamp": f.timestamp,
            "num_annotations": len(f.annotations),
            "num_pending": sum(1 for a in f.annotations if a.status == "pending"),
        }
        for f in job.frames
    ]


@app.get("/jobs/{job_id}/frames/{frame_number}")
def api_get_frame(job_id: str, frame_number: int):
    frame = _require_frame(job_id, frame_number)
    return frame.to_dict()


@app.get("/jobs/{job_id}/frames/{frame_number}/image")
def api_get_frame_image(job_id: str, frame_number: int):
    frame = _require_frame(job_id, frame_number)
    img_path = storage.job_dir(job_id) / frame.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Frame image not found on disk.")
    return FileResponse(img_path)


@app.patch("/jobs/{job_id}/frames/{frame_number}/annotations/{ann_id}")
def api_update_annotation(job_id: str, frame_number: int, ann_id: str, update: AnnotationUpdate):
    job = storage.load_job(job_id)
    frame = _find_frame(job, frame_number)
    ann = _find_annotation(frame, ann_id)

    if update.label is not None:
        ann.label = update.label
    if update.bbox is not None:
        ann.bbox = update.bbox
        ann.source = "human_corrected"
    if update.status is not None:
        if update.status not in ("pending", "approved", "rejected", "edited"):
            raise HTTPException(status_code=400, detail="Invalid status value.")
        ann.status = update.status
        if update.status == "edited":
            ann.source = "human_corrected"

    storage.save_job(job)  # save immediately -> per-frame progress, crash-safe
    return ann.to_dict()


@app.delete("/jobs/{job_id}/frames/{frame_number}/annotations/{ann_id}")
def api_delete_annotation(job_id: str, frame_number: int, ann_id: str):
    job = storage.load_job(job_id)
    frame = _find_frame(job, frame_number)
    ann = _find_annotation(frame, ann_id)
    frame.annotations = [a for a in frame.annotations if a.id != ann.id]
    storage.save_job(job)
    return {"deleted": ann.id}


@app.post("/jobs/{job_id}/frames/{frame_number}/annotations")
def api_add_annotation(job_id: str, frame_number: int, new_ann: AnnotationCreate):
    job = storage.load_job(job_id)
    frame = _find_frame(job, frame_number)
    ann = Annotation(
        label=new_ann.label,
        bbox=new_ann.bbox,
        confidence=1.0,
        source="human_corrected",
        status=new_ann.status,
    )
    frame.annotations.append(ann)
    storage.save_job(job)
    return ann.to_dict()


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

@app.post("/jobs/{job_id}/export")
def api_export(job_id: str, fmt: str = "coco"):
    _require_job(job_id)
    job = storage.load_job(job_id)
    try:
        out_path = export_mod.export(job, fmt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job.status = "done"
    storage.save_job(job)
    return {"job_id": job.job_id, "format": fmt, "path": str(out_path)}


@app.get("/jobs/{job_id}/export/{fmt}/download")
def api_download_export(job_id: str, fmt: str):
    _require_job(job_id)
    job = storage.load_job(job_id)
    out_path = export_mod.export(job, fmt)
    if out_path.is_dir():
        # YOLO export is a directory of .txt files -> zip it on the fly
        zip_base = str(out_path)
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=out_path)
        return FileResponse(zip_path, filename=f"{job_id}_yolo_export.zip")
    return FileResponse(out_path, filename=out_path.name)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _require_job(job_id: str) -> None:
    if not storage.job_exists(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


def _find_frame(job: Job, frame_number: int):
    for f in job.frames:
        if f.frame_number == frame_number:
            return f
    raise HTTPException(status_code=404, detail=f"Frame {frame_number} not found in job {job.job_id}.")


def _require_frame(job_id: str, frame_number: int):
    _require_job(job_id)
    job = storage.load_job(job_id)
    return _find_frame(job, frame_number)


def _find_annotation(frame, ann_id: str) -> Annotation:
    for a in frame.annotations:
        if a.id == ann_id:
            return a
    raise HTTPException(status_code=404, detail=f"Annotation '{ann_id}' not found.")