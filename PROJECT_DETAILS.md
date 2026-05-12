# FaceGuard Pro - Project Documentation & Architecture

This document provides a detailed breakdown of the technology stack and the operational flow of the FaceGuard Pro system, based directly on the current implementation in this repository.

## 1. Technology Stack

### Backend (Python Core)
- **FastAPI**: High-performance web framework used for the REST API and real-time WebSocket streams.
- **SQLite**: Lightweight relational database used via the standard `sqlite3` library to store camera configurations, employee registries, and detection logs.
- **OpenCV (cv2)**: Handles RTSP/USB camera ingestion, image preprocessing (resizing, cropping), and frame annotation (drawing boxes).
- **NumPy**: Used for efficient numerical operations and image data representation.
- **FAISS (Facebook AI Similarity Search)**: The primary vector database and similarity search engine for high-performance facial embedding matching.

### AI & Machine Learning Engine
- **InsightFace (ONNX Runtime)**: The primary face detection and recognition engine. It uses the `buffalo_l` model for high-precision 512-D facial feature extraction.
- **YOLO11 (Ultralytics)**: 
  - **YOLO11-Face**: Used as a fallback detection model if InsightFace is unavailable.
  - **YOLO11-Standard**: Used by the `OpenVocabularyObjectDetector` to identify general objects (bottles, laptops, etc.) and perform person detection.
- **ArcFace**: The underlying deep learning architecture used for generating facial embeddings that remain consistent across different lighting and angles.

### Frontend (React Dashboard)
- **React (Vite)**: Modern component-based UI for the dashboard.
- **WebSockets**: Used for two critical real-time channels:
  1. **Video Stream**: Sends encoded JPEG frames from the backend directly to the browser.
  2. **Event Feed**: Pushes instant notifications when a face is matched.
- **Lucide React**: Vector icons used throughout the interface.

---

## 2. Project Flow & Architecture

The system operates as a multi-threaded pipeline to ensure that AI processing does not block the real-time video display.

### Step 1: Camera Ingestion (`CameraManager.py`)
Each camera added to the system spawns its own **CameraWorker**. This worker runs two parallel threads:
1. **Reader Thread**: Continuously pulls raw frames from the camera URL.
2. **AI Thread**: Every 200ms, it picks the most recent frame and sends it into the AI pipeline.

### Step 2: The Face Recognition Pipeline (`FaceEngine.py`)
When a frame enters the Face Engine, it follows these steps:
1. **Detection**: Locate bounding boxes for all faces in the frame.
2. **Alignment**: Rotate and scale the face so the eyes are level (crucial for accurate matching).
3. **Embedding**: Convert the face image into a **512-dimensional vector** (a list of numbers).
4. **Matching**: Compare this vector against your database using **Cosine Similarity**. If the "distance" is small enough, it's a match.
5. **Temporal Voting**: The system keeps a small buffer of recent results for each person. An identity is only displayed once it has been "voted" on by multiple frames, which prevents the name from flickering.

### Step 3: Object & Person Association (`ObjectDetector.py`)
While faces are being recognized, a parallel process runs the general YOLOv8 model:
1. **Person Tracking**: Identifies all human bodies in the frame.
2. **Identity Mapping**: If a face is recognized within a person's bounding box, that identity is "attached" to the body. This allows the system to keep labeling the person even if they turn their head away from the camera.
3. **Object Interaction**: Detects objects like laptops or bags and records which camera/location they were seen in.

### Step 4: Streaming & Visualization (`Main.py` & `LiveFeed.jsx`)
1. **Frame Encoding**: The backend converts the annotated frame into a JPEG, then a Base64 string.
2. **WS Push**: This string is sent over a WebSocket to the frontend.
3. **Overlay Rendering**: The frontend receives the frame and draws the final "Tactical Orange" boxes as an HTML overlay, ensuring the UI remains responsive and interactive.

### Step 5: Persistence (`Database.py`)
Every confirmed match is saved to the `detection_events` table in SQLite. This includes:
- The employee name and ID.
- The camera location.
- The confidence score.
- A Base64 thumbnail of the detected face for historical review.

---

## 3. Key Model Roles

| Model | Primary Role | Why it is used |
| :--- | :--- | :--- |
| **InsightFace** | Face Recognition | Industry-leading accuracy for identifying registered employees. |
| **YOLO11-Face** | Face Detection | Faster and more robust for finding faces at a distance or in low light. |
| **YOLO11-General** | Object/Person Detection | Highly optimized for real-time tracking of bodies and equipment. |
| **DeepSORT (Logic)** | Tracking | Ensures that "Person A" remains "Person A" even if they temporarily walk behind an object. |
