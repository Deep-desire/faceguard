"""
FaceGuard Pro – Camera Manager
================================
Architecture (3 logical stages, 2 threads):

  Thread-1 (reader):   RTSP → cap.read() as fast as possible
                        → stores latest raw frame in self._raw_frame
  Thread-2 (AI):       wakes every AI_INTERVAL seconds
                        → grabs latest raw frame
                        → runs YOLOv8 + ArcFace pipeline
                        → stores annotated frame + detections
  Async generator:      yields latest annotated (or raw) frame at ~25 fps
                        → WebSocket sends to browser

This decouples video capture from inference so the stream stays smooth even
when CPU inference takes 200-500 ms per frame.
"""

import asyncio
import base64
import logging
import threading
import time
import uuid
from typing import AsyncGenerator, Optional

import cv2
import numpy as np

log = logging.getLogger("faceguard.cameras")

JPEG_QUALITY  = 72          # encode quality
AI_INTERVAL   = 0.20        # seconds between AI frames (~5 fps AI to save CPU)
STREAM_FPS    = 10          # WebSocket delivery fps – keep it low to prevent buffer backlog
STREAM_WIDTH  = 640         # max width for streaming (scale down 1080→640 saves ~65% bandwidth)
MAX_FRAME_KB  = 80          # hard cap: re-encode at lower quality if still too large
MAX_EVENT_Q   = 200


class CameraWorker:
    """
    Non-thread class that owns two daemon threads:
      _reader_thread  – pure RTSP capture loop
      _ai_thread      – periodic AI inference
    """

    def __init__(self, cam_id: str, rtsp_url: str, face_engine, db):
        self.cam_id     = cam_id
        self.rtsp_url   = rtsp_url
        self.face_engine = face_engine
        self.db         = db

        self._stop      = threading.Event()
        self._raw_lock  = threading.Lock()
        self._out_lock  = threading.Lock()

        self._raw_frame: Optional[np.ndarray] = None   # latest from camera
        self._out_frame: Optional[np.ndarray] = None   # latest annotated
        self._detections: list = []
        self._alive     = False

        self._event_q: list = []          # simple list guarded by _out_lock

        # Start threads
        self._reader = threading.Thread(target=self._reader_loop,
                                        daemon=True, name=f"reader-{cam_id[:8]}")
        self._ai     = threading.Thread(target=self._ai_loop,
                                        daemon=True, name=f"ai-{cam_id[:8]}")
        self._reader.start()
        self._ai.start()

    # ── Public ─────────────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        return self._alive and self._reader.is_alive()

    def stop(self):
        self._stop.set()

    def get_latest_frame(self) -> Optional[dict]:
        """Return the most recent annotated frame encoded as base64 JPEG."""
        with self._out_lock:
            frame = self._out_frame
            dets  = list(self._detections)
        if frame is None:
            with self._raw_lock:
                frame = self._raw_frame
            dets = []
        if frame is None:
            return None
        return self._encode(frame, dets)

    def pop_events(self) -> list:
        with self._out_lock:
            events = list(self._event_q)
            self._event_q.clear()
        return events

    # ── Reader thread ───────────────────────────────────────────────────────────

    def _reader_loop(self):
        log.info(f"📸 Reader starting: {self.rtsp_url}")
        reconnect_delay = 3
        while not self._stop.is_set():
            cap = self._open(self.rtsp_url)
            if cap is None:
                time.sleep(reconnect_delay)
                continue
            self._alive = True
            log.info(f"✅ Stream open: {self.cam_id[:8]}")
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                with self._raw_lock:
                    self._raw_frame = frame   # always keep latest
            cap.release()
            self._alive = False
            if not self._stop.is_set():
                log.warning(f"Camera {self.cam_id[:8]} dropped. Reconnecting in {reconnect_delay}s")
                time.sleep(reconnect_delay)
        log.info(f"Reader stopped: {self.cam_id[:8]}")

    def _open(self, url: str) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            log.error(f"Cannot open: {url}")
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ── AI thread ───────────────────────────────────────────────────────────────

    def _ai_loop(self):
        """Run face detection + recognition at a sustainable rate."""
        last_ts = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - last_ts < AI_INTERVAL:
                time.sleep(0.02)
                continue

            with self._raw_lock:
                frame = self._raw_frame
            if frame is None:
                time.sleep(0.05)
                continue

            last_ts = now
            try:
                result    = self.face_engine.process_frame(frame, self.cam_id)
                annotated = result["annotated_frame"]
                detections = result["detections"]

                with self._out_lock:
                    self._out_frame  = annotated
                    self._detections = detections
                    # Collect events for recognized employees
                    for det in detections:
                        if det.get("recognized"):
                            evt = self._make_event(det, annotated)
                            if evt:
                                self._event_q.append(evt)
                                if len(self._event_q) > MAX_EVENT_Q:
                                    self._event_q.pop(0)
            except Exception as e:
                log.error(f"AI error on {self.cam_id[:8]}: {e}", exc_info=False)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _encode(self, frame: np.ndarray, detections: list) -> Optional[dict]:
        # ── Scale down for streaming (most cameras are 1080p+ — too large for WS) ──
        h, w = frame.shape[:2]
        if w > STREAM_WIDTH:
            scale  = STREAM_WIDTH / w
            frame  = cv2.resize(frame, (STREAM_WIDTH, int(h * scale)),
                                interpolation=cv2.INTER_LINEAR)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if len(buf) > MAX_FRAME_KB * 1024:              # still too large → lower quality
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
        b64 = base64.b64encode(buf).decode("utf-8")
        return {
            "type":       "frame",
            "camera_id":  self.cam_id,
            "frame":      b64,
            "detections": detections,
            "timestamp":  time.time(),
        }

    def _make_event(self, det: dict, frame: np.ndarray) -> Optional[dict]:
        try:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                return None
            _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            face_b64 = base64.b64encode(buf).decode("utf-8")
            event = {
                "id":            str(uuid.uuid4()),
                "camera_id":     self.cam_id,
                "employee_id":   det.get("employee_id"),
                "face_id":       det.get("face_id"),
                "employee_name": det.get("name"),
                "confidence":    det.get("confidence"),
                "track_id":      det.get("track_id"),
                "timestamp":     time.time(),
                "frame_b64":     face_b64,
            }
            try:
                self.db.insert_event(event)
            except Exception:
                pass
            return {
                "type":          "detection",
                "camera_id":     self.cam_id,
                "employee_id":   event["employee_id"],
                "face_id":       event["face_id"],
                "employee_name": event["employee_name"],
                "confidence":    event["confidence"],
                "timestamp":     event["timestamp"],
                "face_image":    face_b64,
            }
        except Exception as e:
            log.error(f"Event error: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════

class CameraManager:
    """Manages all active camera workers."""

    def __init__(self, face_engine, db):
        self._face_engine = face_engine
        self._db          = db
        self._workers: dict[str, CameraWorker] = {}
        self._lock    = threading.Lock()

    def start_camera(self, cam_id: str, rtsp_url: str):
        with self._lock:
            if cam_id in self._workers:
                self._workers[cam_id].stop()
            worker = CameraWorker(cam_id, rtsp_url, self._face_engine, self._db)
            self._workers[cam_id] = worker
            log.info(f"Camera {cam_id[:8]} started")

    def stop_camera(self, cam_id: str):
        with self._lock:
            w = self._workers.pop(cam_id, None)
            if w:
                w.stop()
                log.info(f"Camera {cam_id[:8]} stopped")

    def stop_all(self):
        with self._lock:
            for w in self._workers.values():
                w.stop()
            self._workers.clear()

    def is_active(self, cam_id: str) -> bool:
        w = self._workers.get(cam_id)
        return w is not None and w.is_alive()

    async def frame_generator(self, cam_id: str) -> AsyncGenerator[dict, None]:
        """
        Yields frames indefinitely.
        • Active camera  → up to STREAM_FPS, skips if no new annotated frame
        • Inactive camera → status ping every 2 s (keeps WS alive)
        """
        interval   = 1.0 / STREAM_FPS
        last_ts    = 0.0    # timestamp of last sent frame

        while True:
            now    = time.time()
            worker = self._workers.get(cam_id)

            if worker and worker.is_alive():
                frame_data = worker.get_latest_frame()
                if frame_data and now - last_ts >= interval:
                    last_ts = now
                    yield frame_data
                else:
                    await asyncio.sleep(interval / 2)
            else:
                yield {
                    "type":      "status",
                    "camera_id": cam_id,
                    "status":    "inactive",
                    "message":   "Camera not streaming",
                    "timestamp": now,
                }
                await asyncio.sleep(2.0)

    async def get_latest_event(self) -> Optional[dict]:
        for worker in list(self._workers.values()):
            events = worker.pop_events()
            if events:
                return events[0]
        return None
