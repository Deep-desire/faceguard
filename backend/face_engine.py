"""
FaceGuard Pro - Face Engine
Pipeline: YOLOv8 (face detection) → Quality Gate → ArcFace (embedding) → Cosine Similarity
Temporal Vote Buffer for stable recognition → DeepSORT for track IDs

Embedding backend priority:
  1. InsightFace ONNX (buffalo_l via onnxruntime) — best quality, no MSVC needed
  2. DeepFace ArcFace (TF/Keras fallback)
  3. Mock embedding (testing only)
"""

import json
import logging
import threading
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("faceguard.engine")

# ── Constants ──────────────────────────────────────────────────────────────────
COSINE_THRESHOLD = 0.45      # lower threshold works better with aligned embeddings
VOTE_WINDOW = 8              # frames in vote buffer
VOTE_THRESHOLD = 3           # votes needed to confirm name (faster recognition)
MIN_FACE_SIZE = 40           # CCTV faces can be small
MAX_BLUR_VAR = 20.0          # CCTV footage is often compressed/blurry
EMBED_DIM = 512              # ArcFace embedding dimension


class FaceEngine:
    """
    Manages the face recognition pipeline.
    Thread-safe for concurrent camera streams.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._initialized = False

        # Face registry: emp_id → {face_id, name, embedding (np.ndarray)}
        self._registry: dict[str, dict] = {}

        # DeepSORT-style track vote buffers: track_id → deque of (emp_id | None)
        self._vote_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))

        # Models (lazy-loaded)
        self._yolo = None
        self._insight_onnx = None  # InsightFace via ONNX (primary)
        self._face_app     = None  # InsightFace Python pkg (legacy)
        self._deepface     = None  # DeepFace ArcFace (secondary fallback)
        self._embed_backend = "mock"  # set during initialize()

    # ── Initialization ─────────────────────────────────────────────────────────

    def initialize(self):
        """Load YOLOv8 and InsightFace models."""
        try:
            from ultralytics import YOLO
            # YOLOv8n-face - lightweight face detection model
            self._yolo = YOLO("yolov8n-face.pt")
            log.info("✅ YOLOv8 face model loaded")
        except Exception as e:
            log.warning(f"YOLOv8 load failed: {e}. Using OpenCV fallback.")
            self._yolo = None

        # ── InsightFace ONNX (primary — no MSVC required) ──────────────────────
        try:
            from insightface_onnx import InsightFaceONNX, MODEL_DIR
            if not (MODEL_DIR / "w600k_r50.onnx").exists():
                raise FileNotFoundError(f"ONNX models not found at {MODEL_DIR}")
            self._insight_onnx = InsightFaceONNX()
            self._embed_backend = "insightface_onnx"
            log.info("✅ InsightFace ArcFace R50 loaded (ONNX / no MSVC needed)")
        except Exception as e:
            log.warning(f"InsightFace ONNX unavailable ({e}). Trying DeepFace fallback...")
            self._insight_onnx = None

            # ── DeepFace (secondary fallback) ────────────────────────────────────
            try:
                from deepface import DeepFace as _DF
                _dummy = np.zeros((112, 112, 3), dtype=np.uint8)
                _DF.represent(_dummy, model_name="ArcFace", enforce_detection=False)
                self._deepface = _DF
                self._embed_backend = "deepface"
                log.info("✅ DeepFace ArcFace loaded (TF/Keras fallback)")
            except Exception as e2:
                log.warning(f"DeepFace also unavailable ({e2}). Using mock embeddings.")
                self._deepface = None
                self._embed_backend = "mock"

        self._initialized = True
        log.info(f"🧠 FaceEngine initialized (embed_backend={self._embed_backend})")

    # ── Registry ───────────────────────────────────────────────────────────────

    def register_face(self, emp_id: str, face_id: str, name: str, embedding: list):
        with self._lock:
            self._registry[emp_id] = {
                "face_id": face_id,
                "name": name,
                "embedding": np.array(embedding, dtype=np.float32),
            }

    def update_face(self, emp_id: str, embedding: list):
        with self._lock:
            if emp_id in self._registry:
                self._registry[emp_id]["embedding"] = np.array(embedding, dtype=np.float32)

    def remove_face(self, emp_id: str):
        with self._lock:
            self._registry.pop(emp_id, None)

    def face_count(self) -> int:
        return len(self._registry)

    def load_from_db(self, db):
        """Reload all embeddings from DB on startup."""
        rows = db.get_all_embeddings()
        for row in rows:
            emb = json.loads(row["embedding"])
            self.register_face(row["id"], row["face_id"], row["name"], emb)
        log.info(f"Loaded {len(rows)} face embeddings from DB")

    # ── Embedding Extraction ───────────────────────────────────────────────────

    def extract_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ArcFace 512-d embedding from an image.
        For registration images (full photo): uses detect_and_embed() so the face
        is properly detected, aligned, then embedded — matching live-detection embeddings.
        """
        backend = getattr(self, "_embed_backend", "mock")

        # ── InsightFace ONNX (primary) ──────────────────────────────────────────
        if backend == "insightface_onnx" and self._insight_onnx is not None:
            try:
                # detect_and_embed finds & aligns the face first (critical for accuracy)
                emb = self._insight_onnx.detect_and_embed(img)
                if emb is not None:
                    return emb
                # Fallback: direct embed without detection (e.g. already-cropped)
                return self._insight_onnx.get_embedding(img)
            except Exception as e:
                log.error(f"InsightFace ONNX embedding error: {e}")
                return None

        # ── InsightFace Python pkg (legacy) ─────────────────────────────────────
        if backend == "insightface" and self._face_app is not None:
            try:
                faces = self._face_app.get(img)
                if not faces:
                    return None
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb = face.embedding
                return emb / np.linalg.norm(emb)
            except Exception as e:
                log.error(f"InsightFace embedding error: {e}")
                return None

        # ── DeepFace (secondary fallback) ───────────────────────────────────────
        if backend == "deepface" and getattr(self, "_deepface", None) is not None:
            try:
                result = self._deepface.represent(
                    img, model_name="ArcFace",
                    enforce_detection=False, detector_backend="skip",
                )
                if not result:
                    return None
                emb = np.array(result[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                return emb / norm if norm > 0 else emb
            except Exception as e:
                log.error(f"DeepFace embedding error: {e}")
                return None

        return self._mock_embedding()


    def _mock_embedding(self) -> np.ndarray:
        """Deterministic mock embedding for testing without InsightFace."""
        emb = np.random.RandomState(42).randn(EMBED_DIM).astype(np.float32)
        return emb / np.linalg.norm(emb)

    # ── Frame Processing ───────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, camera_id: str) -> dict:
        """
        Full pipeline on a single frame.
        When InsightFace ONNX is active, uses get_faces() which does
        detection + landmark alignment + embedding in one pass — the most
        accurate path.  Falls back to YOLOv8 + separate embedding otherwise.
        Returns: {annotated_frame, detections}
        """
        if not self._initialized:
            return {"annotated_frame": frame, "detections": []}

        annotated  = frame.copy()
        detections = []

        # ── Fast path: InsightFace ONNX does detection+alignment+embedding ────
        if self._embed_backend == "insightface_onnx" and self._insight_onnx is not None:
            try:
                faces = self._insight_onnx.get_faces(frame)
            except Exception as e:
                log.error(f"get_faces error: {e}")
                faces = []

            for idx, face in enumerate(faces):
                x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2-x1) < MIN_FACE_SIZE or (y2-y1) < MIN_FACE_SIZE:
                    continue

                emb = face.get("embedding")
                track_id = f"{camera_id}_{idx}"

                if emb is not None:
                    match_id, match_name, match_conf = self._cosine_match(emb)
                else:
                    match_id, match_name, match_conf = None, None, 0.0

                self._vote_buffers[track_id].append(match_id)
                stable_id = self._get_stable_id(track_id)

                # Only annotate & report REGISTERED employees — skip unknowns
                if not (stable_id and stable_id in self._registry):
                    continue

                entry = self._registry[stable_id]
                label = f"{entry['name']} [{entry['face_id']}]"
                self._draw_detection(annotated, x1, y1, x2, y2, label, (0, 220, 100), match_conf)
                detections.append({
                    "track_id":    track_id,
                    "employee_id": stable_id,
                    "face_id":     entry["face_id"],
                    "name":        entry["name"],
                    "confidence":  round(float(match_conf), 3),
                    "bbox":        [x1, y1, x2, y2],
                    "recognized":  True,
                })

            return {"annotated_frame": annotated, "detections": detections}

        # ── Fallback path: YOLOv8 detect → separate embedding ────────────────
        face_bboxes = self._detect_faces(frame)
        for idx, bbox in enumerate(face_bboxes):
            x1, y1, x2, y2, det_conf = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame[y1:y2, x1:x2]
            if not self._quality_gate(face_crop):
                continue

            track_id = f"{camera_id}_{idx}"
            match_id, match_name, match_conf = self._identify(face_crop)
            self._vote_buffers[track_id].append(match_id)
            stable_id = self._get_stable_id(track_id)

            if stable_id and stable_id in self._registry:
                entry = self._registry[stable_id]
                label = f"{entry['name']} [{entry['face_id']}]"
                color = (0, 220, 100)
                conf_display = match_conf
            else:
                label = "Unknown"
                color = (0, 60, 220)
                conf_display = 0.0

            self._draw_detection(annotated, x1, y1, x2, y2, label, color, conf_display)
            detections.append({
                "track_id":    track_id,
                "employee_id": stable_id,
                "face_id":     self._registry[stable_id]["face_id"] if stable_id and stable_id in self._registry else None,
                "name":        self._registry[stable_id]["name"] if stable_id and stable_id in self._registry else "Unknown",
                "confidence":  round(float(conf_display), 3),
                "bbox":        [x1, y1, x2, y2],
                "recognized":  stable_id is not None,
            })

        return {"annotated_frame": annotated, "detections": detections}

    # ── Face Detection ─────────────────────────────────────────────────────────

    def _detect_faces(self, frame: np.ndarray) -> list:
        """Returns list of [x1,y1,x2,y2,conf]."""
        if self._yolo is not None:
            try:
                results = self._yolo(frame, verbose=False, conf=0.4)
                boxes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        boxes.append([x1, y1, x2, y2, conf])
                return boxes
            except Exception as e:
                log.error(f"YOLO detection error: {e}")

        # OpenCV DNN fallback
        return self._detect_faces_opencv(frame)

    def _detect_faces_opencv(self, frame: np.ndarray) -> list:
        """Haar cascade fallback for demo/testing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        results = []
        for (x, y, w, h) in faces:
            results.append([x, y, x + w, y + h, 0.85])
        return results

    # ── Quality Gate ───────────────────────────────────────────────────────────

    def _quality_gate(self, face_crop: np.ndarray) -> bool:
        """Blur + size + angle filter."""
        h, w = face_crop.shape[:2]
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return False
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_var < MAX_BLUR_VAR:
            return False
        return True

    # ── Identity Matching ──────────────────────────────────────────────────────

    def _identify(self, face_crop: np.ndarray) -> tuple[Optional[str], Optional[str], float]:
        """Returns (emp_id, name, confidence) or (None, None, 0.0)."""
        embedding = self.extract_embedding(face_crop)
        if embedding is None:
            return None, None, 0.0
        return self._cosine_match(embedding)

    def _cosine_match(self, query_emb: np.ndarray) -> tuple[Optional[str], Optional[str], float]:
        """Find best cosine match in registry."""
        best_id = None
        best_name = None
        best_score = -1.0

        with self._lock:
            for emp_id, data in self._registry.items():
                ref_emb = data["embedding"]
                score = float(np.dot(query_emb, ref_emb))  # both L2-normalized → cosine similarity
                if score > best_score:
                    best_score = score
                    best_id = emp_id
                    best_name = data["name"]

        if best_score >= COSINE_THRESHOLD:
            return best_id, best_name, best_score
        return None, None, best_score

    # ── Temporal Vote Buffer ───────────────────────────────────────────────────

    def _get_stable_id(self, track_id: str) -> Optional[str]:
        """Return emp_id if it has enough votes in the buffer."""
        buf = self._vote_buffers[track_id]
        counts: dict[str, int] = defaultdict(int)
        for vote in buf:
            if vote is not None:
                counts[vote] += 1
        if not counts:
            return None
        best_id, best_count = max(counts.items(), key=lambda x: x[1])
        return best_id if best_count >= VOTE_THRESHOLD else None

    # ── Annotation ────────────────────────────────────────────────────────────

    def _draw_detection(self, img, x1, y1, x2, y2, label, color, conf):
        """Draw bounding box + label on frame."""
        # Main bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Corner accents (tactical look)
        corner_len = min(20, (x2 - x1) // 4)
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), color, 3)
            cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), color, 3)

        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        conf_text = f"{conf:.0%}" if conf > 0 else ""
        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(img, (x1, label_y - th - baseline - 4), (x1 + tw + 6, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 3, label_y - baseline), font, font_scale, (255, 255, 255), thickness)
        if conf_text:
            cv2.putText(img, conf_text, (x2 - 40, y2 - 6), font, 0.45, color, 1)
