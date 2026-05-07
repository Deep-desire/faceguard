# FaceGuard Pro 🛡️
### Production CCTV Face Recognition System

Real-time face detection and recognition system for CCTV surveillance using YOLOv8, ArcFace (InsightFace ResNet-100), and DeepSORT tracking.

---

## Architecture

```
faceguard/
├── backend/          # Python FastAPI
│   ├── main.py           # API server, WebSocket endpoints
│   ├── face_engine.py    # YOLOv8 + ArcFace + Vote Buffer pipeline
│   ├── camera_manager.py # RTSP stream management (threaded)
│   ├── database.py       # SQLite persistence layer
│   ├── models.py         # Pydantic request/response models
│   └── requirements.txt
│
└── frontend/         # React app
    ├── src/
    │   ├── App.jsx           # Router + sidebar layout
    │   ├── index.css         # Tactical dark theme
    │   ├── api.js            # Axios + WebSocket helpers
    │   └── pages/
    │       ├── Dashboard.jsx  # Stats + chart + recent events
    │       ├── Cameras.jsx    # Add/edit/delete RTSP cameras
    │       ├── Employees.jsx  # Register employees with face photos
    │       ├── LiveFeed.jsx   # Real-time camera streams + detections
    │       └── Events.jsx     # Detection history table
    └── package.json
```

---

## Detection Pipeline (from diagram)

```
RTSP Frame → Process every 3rd frame
     ↓
Quality Gate (NEW): Blur + size + angle filter
     ↓
ArcFace 512-d embedding (InsightFace ResNet-100)
     ↓
Cosine similarity ≥ 0.55 (NOT Euclidean)
     ↓
Temporal vote buffer (NEW): Name shown after 5/10 frame votes
     ↓
Person track ID (NEW): DeepSORT bounding box track
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (optional, but recommended for real-time performance)

---

## Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 face model (auto-downloads on first run)
# InsightFace models are optional; if installed, they auto-download on first run (~500MB)

# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --ws-ping-interval 0 --ws-ping-timeout 0
```

Important: always start the backend from the activated `backend/venv`. If you run
plain `uvicorn` from a global Python install, it can pick up NumPy 2.x and break
OpenCV with the exact error you saw.

On Windows, prefer `start.ps1` because it already disables websocket keepalive
pings and avoids the `keepalive ping failed` assertion during live camera streaming.

On Windows, the default install now uses OpenCV + YOLO only and the app starts in
fallback face-embedding mode. To enable full ArcFace embeddings, install
`insightface` in an environment that has Microsoft C++ Build Tools available, or
run the backend in Docker/Linux.

The backend will:
1. Initialize SQLite database (`faceguard.db`)
2. Download & load YOLOv8n-face model
3. Download & load InsightFace ArcFace (buffalo_l)
4. Start listening on port 8000

**API Docs**: http://localhost:8000/docs

---

## Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Opens at http://localhost:3000

---

## Usage Guide

### 1. Add Cameras
- Go to **Cameras** tab
- Click **Add Camera**
- Enter name, RTSP URL, and location
- Toggle stream on/off

**RTSP URL formats:**
```
rtsp://username:password@192.168.1.10:554/stream1
rtsp://192.168.1.10:554/h264Preview_01_main
rtsp://admin:admin@192.168.0.100/live.sdp
```

### 2. Register Employees
- Go to **Employees** tab
- Click **Register Employee**
- Fill in name, employee code, department
- Upload **multiple face photos** from different angles (front, left, right, up)
  - More angles = better recognition accuracy
- Face ID is auto-generated: `EMP-{CODE}`

### 3. Monitor Live Feed
- Go to **Live Feed** tab
- Active camera streams appear automatically
- Recognized employees are highlighted with **green bounding boxes + name + Face ID**
- Unknown faces highlighted with **blue bounding boxes**
- Right panel shows real-time detection events

### 4. View History
- Go to **Events** tab
- Filter by camera or recognition status
- View face snapshots, confidence scores, timestamps

---

## API Reference

### Cameras
```
GET    /api/cameras              # List all cameras
POST   /api/cameras              # Add camera
PUT    /api/cameras/{id}         # Update camera
DELETE /api/cameras/{id}         # Delete camera
POST   /api/cameras/{id}/toggle  # Start/stop stream
```

### Employees
```
GET    /api/employees            # List all employees
POST   /api/employees            # Register (multipart: fields + images[])
PUT    /api/employees/{id}       # Update employee
DELETE /api/employees/{id}       # Remove employee
```

### Events
```
GET    /api/events               # Detection history
GET    /api/dashboard/stats      # Dashboard statistics
```

### WebSocket
```
WS /ws/camera/{cam_id}           # Live frame stream (base64 JPEG + detections)
WS /ws/events                    # Real-time detection events
```

---

## Technical Details

| Component | Technology |
|-----------|-----------|
| Face Detection | YOLOv8n-face (Ultralytics) |
| Face Recognition | ArcFace via InsightFace ResNet-100 |
| Embedding | 512-dimensional L2-normalized vectors |
| Similarity | Cosine similarity (threshold: 0.55) |
| Temporal Smoothing | Vote buffer (5/10 frames) |
| Object Tracking | DeepSORT track IDs |
| Backend | FastAPI + uvicorn (async) |
| Database | SQLite (WAL mode) |
| Frontend | React 18 + Recharts |
| Streaming | WebSocket (MJPEG-over-WS) |

---

## Production Deployment

### Docker (recommended)

```bash
# Backend
docker build -t faceguard-backend ./backend
docker run -p 8000:8000 -v ./data:/app/data faceguard-backend

# Frontend  
cd frontend && npm run build
# Serve build/ with nginx
```

### Environment Variables

```bash
# backend/.env
DATABASE_PATH=/data/faceguard.db
UPLOAD_DIR=/data/uploads
COSINE_THRESHOLD=0.55
FRAME_SKIP=3
VOTE_WINDOW=10
VOTE_THRESHOLD=5
```

---

## Performance Tips

- Use GPU for InsightFace: change `ctx_id=-1` to `ctx_id=0` in `face_engine.py`
- Reduce `JPEG_QUALITY` in `camera_manager.py` for lower bandwidth
- Increase `FRAME_SKIP` for lower CPU usage
- Run multiple camera workers on a machine with 8+ cores

---

## License
MIT — Built for production surveillance systems.
