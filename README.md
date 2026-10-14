# Annotator Tool

A lightweight, local annotation tool for reviewing YOLO-generated object detections on videos or image batches. The app was designed for a single operator who wants to upload media, let a model propose boxes, quickly review edge cases, and export clean annotation data in common formats.

This project is currently an MVP-style workflow implementation. It is functional for local annotation work, but it still has several product and engineering gaps to address before it is production-ready.

## Project status

Status: working prototype / MVP

Current implementation includes:
- video upload and frame extraction
- image-batch job creation
- YOLO detection with confidence thresholding
- per-frame review UI
- approve / reject / edit / delete / manual add operations
- export to COCO, YOLO, and JSON
- local job persistence on disk

What is still left to do is documented later in this README under Remaining work.

## What the app does

The UI and backend together support this flow:

1. Create a job from a video or image upload
2. Extract frames from the source media
3. Run YOLO detection over those frames
4. Review any uncertain boxes in the Streamlit interface
5. Approve or edit detections
6. Export only accepted annotations

This reduces labeling effort by letting the model do the first pass and only asking the human to confirm uncertain or missing detections.

## Tech stack

- Python
- FastAPI for the backend API
- Streamlit for the review UI
- OpenCV for frame extraction
- Ultralytics YOLO models for detection
- Pillow for image drawing in the UI
- JSON files for local job storage

## Project structure

```text
annotator_tool/
├── README.md
├── requirements.txt
├── yolo11n.pt
├── yolo11s.pt
├── yolo11m.pt
├── yolo11x.pt
├── yolov8n.pt
├── backend/
│   ├── __init__.py
│   ├── detection.py
│   ├── export.py
│   ├── main.py
│   ├── models.py
│   ├── storage.py
│   └── video_processing.py
├── jobs/
│   ├── <job_id>/
│   │   ├── job.json
│   │   ├── video/
│   │   ├── frames/
│   │   └── exports/
├── ui/
│   └── app.py
└── venv/   # local virtual environment, if created
```

## Features already implemented

### Job creation
- create video-based jobs
- create image-batch jobs
- save metadata and generated frames in a job folder

### Frame processing
- extract frames from video using a configurable interval
- store extracted frame images under each job folder

### Detection
- run YOLO detection on each frame
- set detection confidence threshold
- auto-approve boxes above a configurable threshold
- leave low-confidence boxes in a pending/review state

### Review workflow
- move between frames
- jump to next frame needing review
- approve, reject, edit, and delete boxes
- add new boxes manually
- hide/show label clutter in review mode

### Export
- export accepted annotations as:
  - COCO JSON
  - YOLO label format
  - generic JSON

### Persistence
- each job is saved as JSON in a folder under jobs/
- the app saves progress along the way rather than only at the end

## Setup

### Prerequisites
- Python 3.10 to 3.12
- pip
- internet access for the first model download
- roughly 1 GB of free disk space for dependencies and model weights

### Create and activate virtual environment

macOS / Linux:

```bash
cd annotator_tool
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd annotator_tool
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the project

Start the backend in one terminal:

```bash
source venv/bin/activate
uvicorn backend.main:app --reload
```

Start the Streamlit interface in another terminal:

```bash
source venv/bin/activate
streamlit run ui/app.py
```

Then open the URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Typical workflow

1. Open the app and choose a source type: Video or Images
2. Enter a client name and upload media
3. Choose a frame interval for video sampling
4. Run YOLO detection
5. Review frames and fix boxes that are wrong or missing
6. Export final labels in the desired format

## Data model summary

Jobs are represented in the backend as dataclasses in [backend/models.py](backend/models.py). Each job contains:
- basic job metadata
- a list of frames
- annotations attached to each frame
- status information such as uploaded, processing, in_review, and done

The job state is serialized to JSON and saved under each job folder. This makes it easy to inspect the state and restore progress locally.

## Remaining work / TODOs

This project is usable as a local prototype, but there are still several important tasks left before it is robust enough for broader use.

### High-priority work
- Add unit and integration tests for:
  - frame extraction
  - detection processing
  - export output correctness
  - annotation update logic
- Add validation for bbox values and annotation statuses
- Improve error handling for failed YOLO runs and malformed job files
- Add a better job cleanup and recovery flow for partially broken jobs
- Add logging and structured error reporting

### Product / UX work
- Add a real dashboard view for all jobs and their states
- Improve navigation for large annotation sets
- Add keyboard shortcuts for approve / reject / next frame
- Add drag-to-select or easier box editing experience
- Add a more polished review layout for crowded images

### Performance work
- Support batch processing for many frames more efficiently
- Add optional GPU usage detection and model selection logic
- Optimize repeated image renders in Streamlit
- Consider background job processing for large videos

### Reliability / scalability work
- Replace plain JSON-only storage with a proper database if the tool grows
- Add locking or concurrency safeguards for shared job access
- Add retry logic for network and filesystem failures
- Improve compatibility and stability across OS environments

### Data quality / export work
- Validate COCO output structure against common tooling conventions
- Improve YOLO export naming and class ID consistency
- Add export previews before download
- Support additional annotation types beyond bounding boxes

### Future roadmap ideas
- Support automatic tracking across frames
- Add active learning or model retraining workflows
- Support cloud deployment for remote teams
- Add user authentication and admin roles if multi-user access is needed

## Known limitations

- This is built for a single operator, not a multi-user system
- Storage is local filesystem-based, not database-backed
- There is no test suite yet
- There is no production deployment setup
- The app is not yet hardened for large-scale or heavy enterprise usage

## Future direction

The project is best positioned as a local annotation assistant for small-to-medium labeling workflows. A realistic next step would be to stabilize the MVP, add tests, improve validation, and then expand the workflow with better UX, background processing, and more robust data handling.

If this project is intended to become a client-facing product, the next milestone should be:

1. automated tests
2. more reliable job state handling
3. better review ergonomics
4. export validation and QA checks
5. deployment-ready architecture

## Troubleshooting

### Backend cannot be reached from Streamlit
Make sure the FastAPI server is running in a separate terminal.

```bash
uvicorn backend.main:app --reload
```

### Missing module errors
Make sure you activated the virtual environment and installed dependencies:

```bash
pip install -r requirements.txt
```

### Detection is slow on the first run
The first run downloads model weights and may take longer than usual.

### Windows PowerShell execution policy issue
If activation fails, use Command Prompt or run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## License

This project currently does not include a specific license file. If it is meant for sharing or commercial use, add a license before distribution.
