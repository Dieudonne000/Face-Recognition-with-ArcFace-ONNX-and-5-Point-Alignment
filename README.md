# ArcFace ONNX

An efficient face recognition implementation optimized for CPU-based inference. This system employs the **ArcFace** deep learning architecture through **ONNX Runtime**, integrating **Haar Cascades** for fast face detection and **MediaPipe** for accurate facial alignment.

The project delivers an end-to-end workflow covering face registration through live recognition, complete with a local storage backend and comprehensive testing utilities.

## System architecture

The recognition workflow prioritizes both speed and precision on standard computing hardware:

1. **Detection**: Simultaneous detection of multiple faces using OpenCV Haar Cascades.
2. **Landmarks**: Extraction of 5-point facial landmarks using MediaPipe FaceMesh.
3. **Alignment**: Affine transformation applied to normalize faces to 112x112 pixels.
4. **Embedding**: Generation of feature vectors through the ArcFace ResNet100 ONNX model.
5. **Matching**: Identity comparison performed using Cosine Similarity (with L2 normalization).

## Key features

* **CPU-Focused Performance**: Leverages ONNX Runtime to achieve real-time speeds on standard processors without GPU acceleration.
* **Advanced Pre-processing**: Combines Haar cascades for quick detection with MediaPipe for precise facial alignment.
* **Interactive Enrollment**: Command-line interface that enables users to capture, inspect, and store new face identities.
* **Embedding Storage**: Compact face vector storage implemented using compressed NumPy archives (`.npz` format).
* **Live Visualization**: Real-time video display showing identity names, match confidence, and frame rate statistics.
* **Testing Framework**: Integrated evaluation scripts for measuring system accuracy on test datasets.

## Project structure

```text
face_recognition_arcface_onnx/
├── init_project.py           # Directory initialization script
├── models/
│   └── embedder_arcface.onnx # ArcFace ONNX model file
├── data/
│   ├── db/                   # Face database storage & metadata
│   ├── debug_aligned/        # Alignment visualization output
│   └── enroll/               # Enrollment images organized by identity
├── src/
│   ├── align.py              # Affine transformation implementation
│   ├── camera.py             # Multi-threaded camera handler
│   ├── detect.py             # Face detection interface
│   ├── embed.py              # ONNX model inference handler
│   ├── enroll.py             # Enrollment application main entry
│   ├── evaluate.py           # Dataset evaluation utility
│   ├── haar_5pt.py           # Haar cascade implementation
│   ├── landmarks.py          # MediaPipe FaceMesh helper
│   └── recognize.py          # Recognition application main entry
└── requirements.txt          # Python package dependencies

```

## Installation

### 1. Clone the Repository

Make sure all project files are available in your working directory.

### 2. Initialize Environment

Execute the initialization script to create required data folders and default files:

```bash
python init_project.py

```

### 3. Install Dependencies

A virtual environment is strongly recommended.

```bash
pip install opencv-python numpy onnxruntime mediapipe

```

### 4. Model Setup

Place the `embedder_arcface.onnx` model file in the `models/` folder.

> **Note**: If the model is missing, download a ResNet100 ArcFace ONNX model and save it to the `models/` directory.

## Usage Guide

### 1. Enrollment

Add new faces to the recognition database.

```bash
python -m src.enroll

```

**Controls:**

| Key | Action |
| --- | --- |
| **SPACE** | Take one face snapshot |
| **A** | Enable/disable **Auto-Capture** |
| **S** | **Save** current enrollment to database |
| **R** | **Reset** session (discard unsaved captures) |
| **Q** | Exit application |

### 2. Real-time Recognition

Launch the recognition system using your webcam feed.

```bash
python -m src.recognize

```

**Controls:**

| Key | Action |
| --- | --- |
| **R** | **Reload** database from storage (hot-reload) |
| **D** | Enable/disable **Debug** visualization (landmarks/alignment) |
| **+** | Raise similarity threshold (More restrictive) |
| **-** | Lower similarity threshold (More permissive) |
| **Q** | Exit application |

### 3. Evaluation

Test the system's performance on a validation dataset.

```bash
python -m src.evaluate

```

## Technical Details

* **Input Resolution**: 112x112 pixels (RGB format).
* **Output Vector**: 512-dimensional float embedding.
* **Distance Metric**: Cosine Distance.
* *Values near 1.0* = More similar.
* *Values near 0.0* = Less similar.


* **Storage**: The face database is saved to `data/db/face_db.npz`. This enables quick read/write operations using NumPy's compressed binary format.

## Troubleshooting

* **Camera Access**: Verify that your terminal has webcam permissions (frequently problematic on macOS systems).
* **Import Errors**: If `mediapipe` installation fails, the system will attempt to use a simpler alignment method, though this reduces accuracy.
* **False Positives**: When unknown individuals are incorrectly recognized, press **`+`** during recognition to raise the confidence threshold.
* **Performance**: If you experience slowdowns, confirm you're using `onnxruntime` (CPU version) instead of `onnxruntime-gpu` unless CUDA drivers are properly configured.
