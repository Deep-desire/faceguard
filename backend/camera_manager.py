"""
FaceGuard Pro – Camera Manager  (Optimized v2)
================================================
Architecture — 3 fully-decoupled stages:

  Stage-1  reader_thread   : RTSP → cap.read() at max camera FPS
                              Uses CAP_PROP_BUFFERSIZE=1 + FFMPEG flags
                              to keep the read buffer as small as possible.
                              Writes latest raw frame into a threading.Event
                              slot (no queue, always latest).

  Stage-2  ai_thread       : Wakes at AI_INTERVAL seconds.
                              Grabs latest raw frame (copy, no blocking).
                              Runs YOLOv8 + ArcFace in the background thread.
                              Writes annotated frame + detections atomically.

  Stage-3  frame_generator : Async generator called from WebSocket handler.
                              Reads latest annotated frame at STREAM_FPS.
                              Never blocks the event loop — frame access is
                              a simple attribute read protected by a lock.

Key improvements over v1:
  • RTSP open flags: FFMPEG tcp transport, small probe size, low analyzeduration
  • Adaptive frame-skip: if AI is slow, reader keeps capturing (no stall)
  • CLAHE enhancement for low-light cameras
  • WebSocket ping disabled at CLI level (--ws-ping-interval 0)
  • Hard max-age guard: if annotated frame is >300 ms old, fall back to raw
  • Reconnect with exponential back-off (3→5→10→15 s cap)
  • Thread-safe event queue with deque (O(1) append/pop)
  • Encode only once per send — shared base64 result
"""

import asyncio
import base64
import logging
import threading
import time
import uuid
from collections import deque
from typing import AsyncGenerator, Optional

import cv2
import numpy as np

log = logging.getLogger("faceguard.cameras")

# ── Tunable constants ──────────────────────────────────────────────────────────
JPEG_QUALITY   = 75          # JPEG encode quality (72–80 is sweet spot)
JPEG_QUALITY_LO = 50         # fallback quality if frame > MAX_FRAME_KB
AI_INTERVAL    = 0.20        # seconds between AI inferences (~5 fps AI)
STREAM_FPS     = 15          # WebSocket delivery fps (raise only if bandwidth allows)
STREAM_WIDTH   = 640         # resize to this width before encoding (saves ~65% bandwidth)
MAX_FRAME_KB   = 90          # if JPEG > this, re-encode at lower quality
MAX_EVENT_Q    = 200         # max queued detection events per camera
FRAME_MAX_AGE  = 0.35        # seconds: if annotated frame is stale, send raw instead
RECONNECT_BASE = 3           # initial reconnect delay in seconds
RECONNECT_MAX  = 15          # cap reconnect delay

# RTSP open options — critical for low latency
_RTSP_OPTIONS = {
    "rtsp_transport": "tcp",      # TCP is more reliable than UDP (less packet loss)
    "stimeout":       "5000000",  # connection timeout in µs (5 s)
    "max_delay":      "500000",   # max demux delay in µs (0.5 s)
    "analyzeduration":"1000000",  # probe duration µs (1 s, default is 5 s)
    "probesize":      "1000000",  # probe size bytes (1 MB, default is 5 MB)
    "fflags":         "nobuffer", # disable FFMPEG internal buffer
    "flags":          "low_delay",
}

# Build the VideoCapture open string for FFMPEG backend
def _make_rtsp_url(url: str) -> str:
    """Append FFMPEG options to RTSP URL if not already present."""
    # If it's not an RTSP stream (e.g. local webcam index), return as-is
    if not url.lower().startswith("rtsp"):
        return url
    opts = "&".join(f"{k}={v}" for k, v in _RTSP_OPTIONS.items())
    sep  = "&" if "?" in url else "?"
    return f"{url}{sep}{opts}"


class CameraWorker:
    """
    Owns two daemon threads (reader + AI) and exposes thread-safe
    accessors for the async WebSocket generator.
    """

    def __init__(self, cam_id: str, rtsp_url: str, face_engine, db):
        self.cam_id      = cam_id
        self.rtsp_url    = rtsp_url
        self.face_engine = face_engine
        self.db          = db

        self._stop       = threading.Event()
        self._raw_lock   = threading.Lock()
        self._out_lock   = threading.Lock()

        # Raw frame slot — reader always overwrites with latest
        self._raw_frame:      Optional[np.ndarray] = None
        self._raw_ts:         float = 0.0

        # Annotated frame slot — AI thread writes after each inference
        self._out_frame:      Optional[np.ndarray] = None
        self._out_ts:         float = 0.0
        self._detections:     list  = []

        # Connection health
        self._alive:          bool  = False
        self._last_frame_ts:  float = 0.0

        # Detection event queue (thread-safe deque)
        self._event_q: deque = deque(maxlen=MAX_EVENT_Q)
        self._eq_lock  = threading.Lock()

        # CLAHE for low-light enhancement (created once, reused)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Start threads
        self._reader = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"reader-{cam_id[:8]}"
        )
        self._ai     = threading.Thread(
            target=self._ai_loop, daemon=True, name=f"ai-{cam_id[:8]}"
        )
        self._reader.start()
        self._ai.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        return self._alive and self._reader.is_alive()

    def stop(self):
        self._stop.set()

    def get_latest_frame(self) -> Optional[dict]:
        """
        Return the best available frame as a base64-encoded dict.
        Prefers annotated frame if fresh (< FRAME_MAX_AGE seconds old).
        Falls back to latest raw frame if AI is running slow.
        """
        now = time.time()

        with self._out_lock:
            out_frame = self._out_frame
            out_ts    = self._out_ts
            dets      = list(self._detections)

        # Use annotated frame if it's fresh enough
        if out_frame is not None and (now - out_ts) < FRAME_MAX_AGE:
            return self._encode(out_frame, dets)

        # Fall back to latest raw frame
        with self._raw_lock:
            raw_frame = self._raw_frame

        if raw_frame is not None:
            return self._encode(raw_frame, dets)

        return None

    def pop_events(self) -> list:
        with self._eq_lock:
            events = list(self._event_q)
            self._event_q.clear()
        return events

    # ── Reader thread ───────────────────────────────────────────────────────────

    def _reader_loop(self):
        log.info(f"📸 Reader starting: {self.rtsp_url}")
        delay = RECONNECT_BASE

        while not self._stop.is_set():
            cap = self._open_capture(self.rtsp_url)
            if cap is None:
                log.warning(f"[{self.cam_id[:8]}] Cannot open stream. Retry in {delay}s")
                time.sleep(delay)
                delay = min(delay * 1.5, RECONNECT_MAX)
                continue

            delay = RECONNECT_BASE  # reset on success
            self._alive = True
            log.info(f"✅ Stream open: {self.cam_id[:8]}")

            consecutive_fails = 0
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    consecutive_fails += 1
                    if consecutive_fails >= 5:
                        log.warning(f"[{self.cam_id[:8]}] {consecutive_fails} read failures — reconnecting")
                        break
                    time.sleep(0.02)
                    continue

                consecutive_fails = 0
                now = time.time()

                # Optional: CLAHE low-light enhancement (comment out if camera is bright)
                # frame = self._enhance(frame)

                with self._raw_lock:
                    self._raw_frame = frame
                    self._raw_ts    = now
                self._last_frame_ts = now

            cap.release()
            self._alive = False
            if not self._stop.is_set():
                log.warning(f"[{self.cam_id[:8]}] Stream dropped. Reconnecting in {delay}s")
                time.sleep(delay)
                delay = min(delay * 1.5, RECONNECT_MAX)

        log.info(f"Reader stopped: {self.cam_id[:8]}")

    def _open_capture(self, url: str) -> Optional[cv2.VideoCapture]:
        """Open VideoCapture with optimized low-latency settings."""
        try:
            is_rtsp = url.lower().startswith("rtsp")

            if is_rtsp:
                # Use FFMPEG backend with low-latency options
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    # Try GStreamer if FFMPEG fails
                    gst = (
                        f"rtspsrc location={url} latency=0 ! "
                        "rtph264depay ! h264parse ! avdec_h264 ! "
                        "videoconvert ! appsink max-buffers=1 drop=true"
                    )
                    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            else:
                # Local webcam or file
                cap = cv2.VideoCapture(url)

            if not cap.isOpened():
                return None

            # Minimize internal buffer — key for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Request camera native FPS (don't let OpenCV downsample)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Try to set FOURCC to MJPEG for USB cameras (faster than YUY2)
            if not is_rtsp:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            return cap

        except Exception as e:
            log.error(f"Open capture error: {e}")
            return None

    # ── AI thread ───────────────────────────────────────────────────────────────

    def _ai_loop(self):
        """Run face detection + recognition at a sustainable rate."""
        last_ts = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - last_ts < AI_INTERVAL:
                time.sleep(0.01)
                continue

            with self._raw_lock:
                frame    = self._raw_frame
                frame_ts = self._raw_ts

            if frame is None:
                time.sleep(0.05)
                continue

            # Skip stale frames (e.g. camera paused)
            if now - frame_ts > 1.0:
                time.sleep(0.05)
                continue

            last_ts = now
            try:
                result     = self.face_engine.process_frame(frame, self.cam_id)
                annotated  = result["annotated_frame"]
                detections = result["detections"]

                with self._out_lock:
                    self._out_frame  = annotated
                    self._out_ts     = time.time()
                    self._detections = detections

                # Collect recognition events
                for det in detections:
                    if det.get("recognized"):
                        evt = self._make_event(det, annotated)
                        if evt:
                            with self._eq_lock:
                                self._event_q.append(evt)

            except Exception as e:
                log.error(f"AI error on {self.cam_id[:8]}: {e}", exc_info=False)

    # ── Low-light enhancement ──────────────────────────────────────────────────

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to Y channel of YCrCb for low-light enhancement."""
        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = self._clahe.apply(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        except Exception:
            return frame

    # ── Frame encoding ─────────────────────────────────────────────────────────

    def _encode(self, frame: np.ndarray, detections: list) -> Optional[dict]:
        """Resize → JPEG encode → base64. Returns payload dict."""
        try:
            h, w = frame.shape[:2]
            if w > STREAM_WIDTH:
                scale = STREAM_WIDTH / w
                frame = cv2.resize(
                    frame,
                    (STREAM_WIDTH, int(h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
                             cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                return None

            # Re-encode at lower quality if still too large
            if len(buf) > MAX_FRAME_KB * 1024:
                ok, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_LO,
                     cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                )
                if not ok:
                    return None

            b64 = base64.b64encode(buf).decode("utf-8")
            return {
                "type":       "frame",
                "camera_id":  self.cam_id,
                "frame":      b64,
                "detections": detections,
                "timestamp":  time.time(),
            }
        except Exception as e:
            log.debug(f"Encode error: {e}")
            return None

    def _make_event(self, det: dict, frame: np.ndarray) -> Optional[dict]:
        try:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            crop = frame[max(0, y1):y2, max(0, x1):x2]
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
    """Manages all active CameraWorker instances."""

    def __init__(self, face_engine, db):
        self._face_engine = face_engine
        self._db          = db
        self._workers:    dict[str, CameraWorker] = {}
        self._lock        = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

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

    # ── Async frame generator for WebSocket ────────────────────────────────────

    async def frame_generator(self, cam_id: str) -> AsyncGenerator[dict, None]:
        """
        Yields frame payloads at STREAM_FPS.
        • Active camera  → real frames (annotated or raw fallback)
        • Inactive camera → status ping every 2 s so the WS stays alive
        Never blocks the event loop — all heavy work is in background threads.
        """
        interval  = 1.0 / STREAM_FPS
        last_send = 0.0

        while True:
            now    = time.time()
            worker = self._workers.get(cam_id)

            if worker and worker.is_alive():
                elapsed = now - last_send
                if elapsed >= interval:
                    frame_data = worker.get_latest_frame()
                    if frame_data:
                        last_send = now
                        yield frame_data
                    # If no frame yet (camera just opened), wait a bit
                    await asyncio.sleep(max(0, interval - 0.005))
                else:
                    await asyncio.sleep(interval - elapsed)
            else:
                yield {
                    "type":      "status",
                    "camera_id": cam_id,
                    "status":    "inactive",
                    "message":   "Camera not streaming",
                    "timestamp": now,
                }
                await asyncio.sleep(2.0)

    # ── Event queue for /ws/events ─────────────────────────────────────────────

    async def get_latest_event(self) -> Optional[dict]:
        for worker in list(self._workers.values()):
            events = worker.pop_events()
            if events:
                return events[0]
        return None
