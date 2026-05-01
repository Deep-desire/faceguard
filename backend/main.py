"""
FaceGuard Pro - Production CCTV Face Recognition System
Backend: FastAPI + YOLOv8 + ArcFace (InsightFace ResNet-100) + DeepSORT
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from database import Database
from face_engine import FaceEngine
from camera_manager import CameraManager
from models import (
    CameraCreate, CameraUpdate, EmployeeCreate, EmployeeUpdate,
    Camera, Employee, DetectionEvent
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("faceguard")

# ── Directories ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads" / "faces"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Global instances ────────────────────────────────────────────────────────────
db = Database()
face_engine = FaceEngine()
camera_manager = CameraManager(face_engine, db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 FaceGuard Pro starting...")
    db.init()
    face_engine.initialize()
    face_engine.load_from_db(db)   # ← load employee embeddings into memory

    # Restore cameras that were active when the server last ran
    active_cameras = [c for c in db.get_all_cameras() if c.get("is_active")]
    for cam in active_cameras:
        try:
            camera_manager.start_camera(cam["id"], cam["rtsp_url"])
            log.info(f"📷 Auto-started camera: {cam['name']} ({cam['id'][:8]}...)")
        except Exception as e:
            log.warning(f"Could not auto-start camera {cam['id']}: {e}")

    log.info("✅ All systems ready")
    yield
    log.info("🛑 Shutting down FaceGuard Pro...")
    camera_manager.stop_all()


app = FastAPI(
    title="FaceGuard Pro API",
    description="Production CCTV Face Recognition System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(BASE_DIR / "uploads")), name="uploads")


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/cameras", response_model=list[Camera])
async def list_cameras():
    cameras = db.get_all_cameras()
    for cam in cameras:
        cam["is_streaming"] = camera_manager.is_active(cam["id"])
    return cameras


@app.post("/api/cameras", response_model=Camera, status_code=201)
async def add_camera(payload: CameraCreate):
    cam_id = str(uuid.uuid4())
    camera = {
        "id": cam_id,
        "name": payload.name,
        "rtsp_url": payload.rtsp_url,
        "location": payload.location,
        "is_active": payload.is_active,
        "created_at": time.time(),
    }
    db.insert_camera(camera)
    if payload.is_active:
        camera_manager.start_camera(cam_id, payload.rtsp_url)
    log.info(f"Camera added: {payload.name} ({cam_id})")
    return {**camera, "is_streaming": payload.is_active}


@app.put("/api/cameras/{cam_id}", response_model=Camera)
async def update_camera(cam_id: str, payload: CameraUpdate):
    cam = db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    updates = payload.model_dump(exclude_none=True)
    db.update_camera(cam_id, updates)
    # Restart stream if URL or active state changed
    if "rtsp_url" in updates or "is_active" in updates:
        camera_manager.stop_camera(cam_id)
        new_active = updates.get("is_active", cam["is_active"])
        new_url = updates.get("rtsp_url", cam["rtsp_url"])
        if new_active:
            camera_manager.start_camera(cam_id, new_url)
    updated = db.get_camera(cam_id)
    updated["is_streaming"] = camera_manager.is_active(cam_id)
    return updated


@app.delete("/api/cameras/{cam_id}", status_code=204)
async def delete_camera(cam_id: str):
    if not db.get_camera(cam_id):
        raise HTTPException(404, "Camera not found")
    camera_manager.stop_camera(cam_id)
    db.delete_camera(cam_id)
    log.info(f"Camera deleted: {cam_id}")


@app.post("/api/cameras/{cam_id}/toggle")
async def toggle_camera(cam_id: str):
    cam = db.get_camera(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    if camera_manager.is_active(cam_id):
        camera_manager.stop_camera(cam_id)
        db.update_camera(cam_id, {"is_active": False})
        return {"status": "stopped"}
    else:
        camera_manager.start_camera(cam_id, cam["rtsp_url"])
        db.update_camera(cam_id, {"is_active": True})
        return {"status": "started"}


# ══════════════════════════════════════════════════════════════════════════════
# EMPLOYEE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/employees", response_model=list[Employee])
async def list_employees():
    return db.get_all_employees()


@app.post("/api/employees", response_model=Employee, status_code=201)
async def add_employee(
    name: str = Form(...),
    department: str = Form(...),
    employee_code: str = Form(...),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    images: list[UploadFile] = File(...),
):
    emp_id = str(uuid.uuid4())
    face_id = f"EMP-{employee_code.upper()}"

    # Save uploaded images & extract embeddings
    emp_dir = UPLOAD_DIR / emp_id
    emp_dir.mkdir(parents=True, exist_ok=True)

    embeddings = []
    saved_paths = []

    for idx, img_file in enumerate(images):
        contents = await img_file.read()
        img_path = emp_dir / f"face_{idx}.jpg"
        img_path.write_bytes(contents)
        saved_paths.append(str(img_path.relative_to(BASE_DIR / "uploads")))

        # Extract embedding
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        embedding = face_engine.extract_embedding(img)
        if embedding is not None:
            embeddings.append(embedding.tolist())

    if not embeddings:
        raise HTTPException(400, "No valid face detected in uploaded images")

    # Average embedding for robustness
    avg_embedding = np.mean(embeddings, axis=0).tolist()

    employee = {
        "id": emp_id,
        "face_id": face_id,
        "name": name,
        "department": department,
        "employee_code": employee_code,
        "email": email,
        "phone": phone,
        "image_paths": json.dumps(saved_paths),
        "embedding": json.dumps(avg_embedding),
        "created_at": time.time(),
    }
    db.insert_employee(employee)
    face_engine.register_face(emp_id, face_id, name, avg_embedding)
    log.info(f"Employee registered: {name} | {face_id}")
    return _format_employee(employee)


@app.put("/api/employees/{emp_id}", response_model=Employee)
async def update_employee(
    emp_id: str,
    name: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    images: Optional[list[UploadFile]] = File(None),
):
    emp = db.get_employee(emp_id)
    if not emp:
        raise HTTPException(404, "Employee not found")

    updates = {}
    if name:
        updates["name"] = name
    if department:
        updates["department"] = department
    if email:
        updates["email"] = email
    if phone:
        updates["phone"] = phone

    if images:
        emp_dir = UPLOAD_DIR / emp_id
        emp_dir.mkdir(parents=True, exist_ok=True)
        embeddings = []
        saved_paths = []
        for idx, img_file in enumerate(images):
            contents = await img_file.read()
            img_path = emp_dir / f"face_new_{idx}.jpg"
            img_path.write_bytes(contents)
            saved_paths.append(str(img_path.relative_to(BASE_DIR / "uploads")))
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            embedding = face_engine.extract_embedding(img)
            if embedding is not None:
                embeddings.append(embedding.tolist())

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            updates["embedding"] = json.dumps(avg_embedding)
            updates["image_paths"] = json.dumps(saved_paths)
            face_engine.update_face(emp_id, avg_embedding)

    db.update_employee(emp_id, updates)
    updated = db.get_employee(emp_id)
    return _format_employee(updated)


@app.delete("/api/employees/{emp_id}", status_code=204)
async def delete_employee(emp_id: str):
    if not db.get_employee(emp_id):
        raise HTTPException(404, "Employee not found")
    face_engine.remove_face(emp_id)
    db.delete_employee(emp_id)
    log.info(f"Employee deleted: {emp_id}")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD & DETECTION EVENTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard/stats")
async def dashboard_stats():
    cameras = db.get_all_cameras()
    employees = db.get_all_employees()
    active_cams = sum(1 for c in cameras if camera_manager.is_active(c["id"]))
    events_today = db.get_events_today()
    unique_today = db.get_unique_detections_today()
    return {
        "total_cameras": len(cameras),
        "active_cameras": active_cams,
        "total_employees": len(employees),
        "events_today": events_today,
        "unique_detections_today": unique_today,
        "registered_faces": face_engine.face_count(),
    }


@app.get("/api/events")
async def list_events(limit: int = 100, camera_id: Optional[str] = None):
    return db.get_events(limit=limit, camera_id=camera_id)


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET - Live Feed
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/camera/{cam_id}")
async def camera_stream(websocket: WebSocket, cam_id: str):
    # Disable library-level keepalive pings — we're sending frames continuously
    # and the ping races with frame data, causing buffer assertion errors.
    websocket._send_queue = None   # no-op guard
    await websocket.accept()
    log.info(f"WS connected: camera {cam_id}")
    try:
        async for frame_data in camera_manager.frame_generator(cam_id):
            try:
                payload = json.dumps(frame_data)
                # If the client can't receive within 1.5 s, drop this frame
                await asyncio.wait_for(
                    websocket.send_text(payload),
                    timeout=1.5,
                )
            except asyncio.TimeoutError:
                pass   # frame dropped — client is backlogged, continue
    except WebSocketDisconnect:
        log.info(f"WS disconnected: camera {cam_id}")
    except Exception as e:
        log.debug(f"WS closed for {cam_id}: {type(e).__name__}")
    finally:
        log.info(f"WS session ended: camera {cam_id}")


@app.websocket("/ws/events")
async def events_stream(websocket: WebSocket):
    await websocket.accept()
    log.info("WS events stream connected")
    try:
        while True:
            event = await camera_manager.get_latest_event()
            if event:
                await websocket.send_text(json.dumps(event))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        log.info("WS events stream disconnected")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _format_employee(emp: dict) -> dict:
    paths = json.loads(emp.get("image_paths", "[]"))
    return {
        **emp,
        "image_paths": paths,
        "embedding": None,  # don't expose raw embedding
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ws_ping_interval=None,   # disable keepalive pings — we stream frames continuously
        ws_ping_timeout=None,
    )


def _ensure_numpy_opencv_compat() -> None:
    """
    Fail fast with a clear message if the interpreter is using NumPy 2.x.

    The OpenCV wheel used by this project is built against NumPy 1.x. If the
    backend starts with a different Python environment, OpenCV crashes during
    import with a confusing low-level error. This check turns that into an
    actionable message.
    """
    try:
        numpy_version = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        return

    major = int(numpy_version.split(".", 1)[0])
    if major >= 2:
        raise RuntimeError(
            "Incompatible Python environment detected: NumPy "
            f"{numpy_version} is installed, but FaceGuard requires NumPy<2 "
            "because the OpenCV/InsightFace wheels in this project were built "
            "for NumPy 1.x.\n\n"
            "Fix:\n"
            "  cd backend\n"
            "  .\\venv\\Scripts\\activate\n"
            "  pip install -r requirements.txt\n"
            "  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
        )


_ensure_numpy_opencv_compat()

import cv2
import numpy as np
