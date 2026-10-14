"""
Streamlit review UI for Annotator Tool.

Talks to the FastAPI backend over HTTP (localhost). This is the screen
you actually sit at all day: upload a video, kick off detection, then
step through frames approving / correcting / deleting boxes, then export.

Run with:  streamlit run ui/app.py
(backend must already be running: uvicorn backend.main:app --reload)
"""
from __future__ import annotations

import io
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

API_BASE = os.environ.get("ANNOTATOR_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Annotator Tool", layout="wide")


# ---------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------

def api_list_jobs():
    r = requests.get(f"{API_BASE}/jobs")
    r.raise_for_status()
    return r.json()


def api_create_job(client_name, interval, video_file):
    files = {"video": (video_file.name, video_file.getvalue())}
    data = {"client_name": client_name, "frame_interval_sec": interval}
    r = requests.post(f"{API_BASE}/jobs", data=data, files=files)
    r.raise_for_status()
    return r.json()


def api_create_image_job(client_name, image_files):
    files = [("images", (f.name, f.getvalue())) for f in image_files]
    data = {"client_name": client_name}
    r = requests.post(f"{API_BASE}/jobs/images", data=data, files=files)
    r.raise_for_status()
    return r.json()


def api_get_job(job_id):
    r = requests.get(f"{API_BASE}/jobs/{job_id}")
    r.raise_for_status()
    return r.json()


def api_run_detection(job_id, auto_approve_threshold, weights="yolo11n.pt"):
    r = requests.post(
        f"{API_BASE}/jobs/{job_id}/detect",
        params={"auto_approve_threshold": auto_approve_threshold, "weights": weights},
    )
    r.raise_for_status()
    return r.json()


def api_list_frames(job_id):
    r = requests.get(f"{API_BASE}/jobs/{job_id}/frames")
    r.raise_for_status()
    return r.json()


def api_get_frame(job_id, frame_number):
    r = requests.get(f"{API_BASE}/jobs/{job_id}/frames/{frame_number}")
    r.raise_for_status()
    return r.json()


def api_get_frame_image(job_id, frame_number):
    r = requests.get(f"{API_BASE}/jobs/{job_id}/frames/{frame_number}/image")
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def api_update_annotation(job_id, frame_number, ann_id, **kwargs):
    r = requests.patch(
        f"{API_BASE}/jobs/{job_id}/frames/{frame_number}/annotations/{ann_id}", json=kwargs
    )
    r.raise_for_status()
    return r.json()


def api_delete_annotation(job_id, frame_number, ann_id):
    r = requests.delete(f"{API_BASE}/jobs/{job_id}/frames/{frame_number}/annotations/{ann_id}")
    r.raise_for_status()
    return r.json()


def api_add_annotation(job_id, frame_number, label, bbox):
    r = requests.post(
        f"{API_BASE}/jobs/{job_id}/frames/{frame_number}/annotations",
        json={"label": label, "bbox": bbox, "status": "approved"},
    )
    r.raise_for_status()
    return r.json()


def api_export(job_id, fmt):
    r = requests.post(f"{API_BASE}/jobs/{job_id}/export", params={"fmt": fmt})
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------

STATUS_COLORS = {
    "pending": "#f5a623",
    "approved": "#2ecc71",
    "rejected": "#e74c3c",
    "edited": "#3498db",
}


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_boxes(image: Image.Image, annotations: list, show_pending_only_labels: bool = False) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font_size = max(16, img.width // 55)
    font = _load_font(font_size)

    visible = [a for a in annotations if a["status"] != "rejected"]
    visible.sort(key=lambda a: 0 if a["status"] == "pending" else 1)

    for ann in reversed(visible):
        x, y, w, h = ann["bbox"]
        color = STATUS_COLORS.get(ann["status"], "#f5a623")
        box_width = 4 if ann["status"] == "pending" else 2
        draw.rectangle([x, y, x + w, y + h], outline=color, width=box_width)

        if show_pending_only_labels and ann["status"] != "pending":
            continue

        label_text = f"{ann['label']} {ann['confidence']:.2f}"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad = 4
        label_y = y - text_h - 2 * pad if y - text_h - 2 * pad > 0 else y
        draw.rectangle(
            [x, label_y, x + text_w + 2 * pad, label_y + text_h + 2 * pad],
            fill=color,
        )
        draw.text((x + pad, label_y + pad), label_text, fill="black", font=font)
    return img


# ---------------------------------------------------------------------
# Sidebar: job selection / creation
# ---------------------------------------------------------------------

st.sidebar.title("Annotator Tool")
st.sidebar.caption("Upload → detect → review → export")

try:
    jobs = api_list_jobs()
except requests.exceptions.ConnectionError:
    st.error(
        "Can't reach the backend API. Start it first:\n\n"
        "`uvicorn backend.main:app --reload`"
    )
    st.stop()

st.sidebar.subheader("New job")
source_kind = st.sidebar.radio("Source", ["Video", "Images"], horizontal=True)

if source_kind == "Video":
    with st.sidebar.form("new_job_form", clear_on_submit=True):
        client_name = st.text_input("Client name")
        interval = st.number_input("Frame interval (seconds)", min_value=0.1, value=1.0, step=0.1)
        video_file = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
        submitted = st.form_submit_button("Upload & extract frames")
        if submitted:
            if not client_name or not video_file:
                st.sidebar.error("Client name and video file are both required.")
            else:
                with st.spinner("Extracting frames..."):
                    result = api_create_job(client_name, interval, video_file)
                st.sidebar.success(f"Job created: {result['job_id']} ({result['num_frames']} frames)")
                st.rerun()
else:
    with st.sidebar.form("new_image_job_form", clear_on_submit=True):
        client_name = st.text_input("Client name", key="img_client_name")
        image_files = st.file_uploader(
            "Image files",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Upload images")
        if submitted:
            if not client_name or not image_files:
                st.sidebar.error("Client name and at least one image are both required.")
            else:
                with st.spinner("Uploading images..."):
                    result = api_create_image_job(client_name, image_files)
                st.sidebar.success(f"Job created: {result['job_id']} ({result['num_frames']} images)")
                st.rerun()

st.sidebar.subheader("Existing jobs")
job_labels = {f"{j['client_name']} — {j['job_id']} ({j['status']})": j["job_id"] for j in jobs}
if not job_labels:
    st.sidebar.info("No jobs yet — create one above.")
    st.stop()

selected_label = st.sidebar.selectbox("Select a job", list(job_labels.keys()))
job_id = job_labels[selected_label]
job = api_get_job(job_id)

st.sidebar.metric("Frames", job["num_frames"])
st.sidebar.metric("Status", job["status"])

st.sidebar.subheader("Detection")
model_choice = st.sidebar.selectbox(
    "Model",
    options=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
    index=1,  # default to 's' — noticeably more accurate than nano, still fast on CPU
    help=(
        "n (nano) is fastest but least accurate — most likely to confuse similar "
        "objects (e.g. suitcase vs. handbag). Going up to s/m/l/x trades speed and "
        "download size for noticeably better accuracy. Each size downloads once, "
        "the first time you use it."
    ),
)
auto_approve_on = st.sidebar.checkbox(
    "Auto-approve confident boxes", value=True,
    help="Boxes at or above the confidence threshold are approved automatically. "
         "Only boxes the model is unsure about are left for you to review.",
)
auto_approve_threshold = None
if auto_approve_on:
    auto_approve_threshold = st.sidebar.slider(
        "Confidence threshold", min_value=0.50, max_value=0.99, value=0.85, step=0.01,
    )

if st.sidebar.button("Run YOLO detection", use_container_width=True):
    with st.spinner(f"Running detection with {model_choice}..."):
        det = api_run_detection(job_id, auto_approve_threshold, model_choice)
    st.sidebar.success(
        f"Detected {det['total_boxes']} boxes — "
        f"{det.get('auto_approved', 0)} auto-approved, "
        f"{det.get('needs_review', det['total_boxes'])} need your review."
    )
    st.rerun()

st.sidebar.subheader("Export")
fmt = st.sidebar.selectbox("Format", ["coco", "yolo", "json"])
if st.sidebar.button("Export", use_container_width=True):
    result = api_export(job_id, fmt)
    st.sidebar.success(f"Exported to: {result['path']}")
    download_url = f"{API_BASE}/jobs/{job_id}/export/{fmt}/download"
    st.sidebar.markdown(f"[Download export]({download_url})")


# ---------------------------------------------------------------------
# Main: frame review
# ---------------------------------------------------------------------

st.title(f"Reviewing: {job['client_name']}")

frames = api_list_frames(job_id)
if not frames:
    st.info("No frames yet. Something went wrong with upload/extraction.")
    st.stop()

frame_numbers = [f["frame_number"] for f in frames]

if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0

col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
with col_nav1:
    if st.button("⬅ Previous", use_container_width=True) and st.session_state.current_frame_idx > 0:
        st.session_state.current_frame_idx -= 1
with col_nav3:
    if (
        st.button("Next ➡", use_container_width=True)
        and st.session_state.current_frame_idx < len(frame_numbers) - 1
    ):
        st.session_state.current_frame_idx += 1
with col_nav2:
    st.session_state.current_frame_idx = st.slider(
        "Frame", 0, len(frame_numbers) - 1, st.session_state.current_frame_idx
    )

frames_with_pending = [f["frame_number"] for f in frames if f["num_pending"] > 0]
if frames_with_pending:
    if st.button(
        f"⏭ Jump to next frame needing review ({len(frames_with_pending)} left)",
        use_container_width=True,
    ):
        current_num = frame_numbers[st.session_state.current_frame_idx]
        next_candidates = [n for n in frames_with_pending if n > current_num]
        target = next_candidates[0] if next_candidates else frames_with_pending[0]
        st.session_state.current_frame_idx = frame_numbers.index(target)
        st.rerun()
else:
    st.success("No frames need review — everything's approved, rejected, or edited. ✅")

current_frame_number = frame_numbers[st.session_state.current_frame_idx]
frame_data = api_get_frame(job_id, current_frame_number)
pending_count = sum(1 for a in frame_data["annotations"] if a["status"] == "pending")
st.caption(
    f"Frame {current_frame_number} · t={frame_data['timestamp']}s · "
    f"{len(frame_data['annotations'])} boxes ({pending_count} pending review)"
)

col_img, col_list = st.columns([2, 1])

with col_img:
    labels_mode = st.radio(
        "Labels shown", ["All boxes", "Pending only (less clutter)"], horizontal=True
    )
    image = api_get_frame_image(job_id, current_frame_number)
    annotated = draw_boxes(
        image,
        frame_data["annotations"],
        show_pending_only_labels=(labels_mode == "Pending only (less clutter)"),
    )
    st.image(annotated, use_container_width=True)
    st.caption(
        "🟠 pending (thicker outline, drawn on top) · 🟢 approved · 🔵 edited · "
        "rejected boxes are hidden from the image"
    )

with col_list:
    st.subheader("Boxes")
    for ann in frame_data["annotations"]:
        with st.expander(
            f"{ann['label']} ({ann['confidence']:.2f}) — {ann['status']}", expanded=(ann["status"] == "pending")
        ):
            new_label = st.text_input("Label", value=ann["label"], key=f"label_{ann['id']}")
            x, y, w, h = ann["bbox"]
            c1, c2, c3, c4 = st.columns(4)
            new_x = c1.number_input("x", value=float(x), key=f"x_{ann['id']}")
            new_y = c2.number_input("y", value=float(y), key=f"y_{ann['id']}")
            new_w = c3.number_input("w", value=float(w), key=f"w_{ann['id']}")
            new_h = c4.number_input("h", value=float(h), key=f"h_{ann['id']}")

            b1, b2, b3, b4 = st.columns(4)
            if b1.button("✅ Approve", key=f"approve_{ann['id']}", use_container_width=True):
                api_update_annotation(job_id, current_frame_number, ann["id"], status="approved")
                st.rerun()
            if b2.button("✏️ Save edit", key=f"edit_{ann['id']}", use_container_width=True):
                changed = (
                    new_label != ann["label"]
                    or [new_x, new_y, new_w, new_h] != ann["bbox"]
                )
                api_update_annotation(
                    job_id,
                    current_frame_number,
                    ann["id"],
                    label=new_label,
                    bbox=[new_x, new_y, new_w, new_h],
                    status="edited" if changed else ann["status"],
                )
                st.rerun()
            if b3.button("❌ Reject", key=f"reject_{ann['id']}", use_container_width=True):
                api_update_annotation(job_id, current_frame_number, ann["id"], status="rejected")
                st.rerun()
            if b4.button("🗑 Delete", key=f"delete_{ann['id']}", use_container_width=True):
                api_delete_annotation(job_id, current_frame_number, ann["id"])
                st.rerun()

    st.divider()
    st.subheader("Add box manually")
    with st.form(f"add_box_{current_frame_number}", clear_on_submit=True):
        new_label = st.text_input("Label", key=f"new_label_{current_frame_number}")
        c1, c2, c3, c4 = st.columns(4)
        nx = c1.number_input("x", value=0.0, key=f"new_x_{current_frame_number}")
        ny = c2.number_input("y", value=0.0, key=f"new_y_{current_frame_number}")
        nw = c3.number_input("w", value=50.0, key=f"new_w_{current_frame_number}")
        nh = c4.number_input("h", value=50.0, key=f"new_h_{current_frame_number}")
        if st.form_submit_button("Add box"):
            if new_label:
                api_add_annotation(job_id, current_frame_number, new_label, [nx, ny, nw, nh])
                st.rerun()
            else:
                st.warning("Give the box a label first.")