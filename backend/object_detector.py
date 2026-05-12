"""
Open-vocabulary object detection and stable object identity management.

This module keeps the object branch isolated from the face pipeline so the live
worker can run both branches in parallel and persist object inventory entries
with stable IDs.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("faceguard.objects")

BASE_DIR = Path(__file__).parent
OBJECT_UPLOAD_DIR = BASE_DIR / "uploads" / "objects"
OBJECT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OBJECT_PROMPTS = [
    "backpack",
    "bag",
    "handbag",
    "suitcase",
    "briefcase",
    "laptop",
    "phone",
    "tablet",
    "wallet",
    "keys",
    "bottle",
    "cup",
    "book",
    "box",
    "package",
    "helmet",
    "umbrella",
    "camera",
    "tool",
    "knife",
    "weapon",
    "mic",
]

PERSON_PROMPT = ["person"]


class OpenVocabularyObjectDetector:
    def __init__(self, db, weight_path: Optional[Path] = None):
        self._db = db
        self._weight_path = Path(weight_path) if weight_path else BASE_DIR / "yolo11n.pt"
        self._model = None
        self._initialized = False
        self._track_to_object_id: dict[str, str] = {}
        self._track_to_signature: dict[str, str] = {}
        self._track_last_seen: dict[str, float] = {}
        self._signature_to_object_id: dict[str, str] = {}
        self._signature_last_seen: dict[str, float] = {}
        self._labels = ["person", *OBJECT_PROMPTS]

    def _ensure_model(self):
        if self._initialized:
            return
        from ultralytics import YOLO

        self._model = YOLO(str(self._weight_path))
        try:
            self._model.set_classes(self._labels)
        except Exception:
            log.warning("World model class prompting unavailable; falling back to default classes")
        self._initialized = True
        log.info("✅ Open-vocabulary object detector ready")

    def detect(self, frame: np.ndarray, camera_id: str, face_detections: list[dict]) -> tuple[list[dict], list[dict]]:
        self._ensure_model()
        now = time.time()

        try:
            # Use track with persist=True for stable ID maintenance across frames
            results = self._model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.28,
            )
        except Exception as exc:
            log.error(f"Object detection failed: {exc}")
            return [], []

        persons: list[dict] = []
        objects: list[dict] = []

        if not results:
            return [], []

        result = results[0]
        names = result.names or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return [], []

        for box in boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            label = str(names.get(cls_id, cls_id)).lower()
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox = [x1, y1, x2, y2]
            crop = self._safe_crop(frame, bbox)
            signature = self._appearance_hash(crop) if crop.size else None
            
            # Prefer YOLO's internal tracker ID if available
            yolo_id = str(int(box.id[0])) if box.id is not None else None
            track_id = yolo_id or self._synthetic_track_id(label, bbox, signature)

            if label == "person":
                person = {
                    "kind": "person",
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": conf,
                    "employee_id": None,
                    "employee_name": None,
                    "face_id": None,
                    "recognized": False,
                    "label": "person",
                    "appearance_hash": signature,
                }
                persons.append(person)
                continue

            object_id = self._resolve_object_id(
                camera_id=camera_id,
                object_label=label,
                appearance_hash=signature,
                track_id=track_id,
                employee_id=None,
            )

            objects.append({
                "kind": "object",
                "track_id": track_id,
                "object_id": object_id,
                "bbox": bbox,
                "confidence": conf,
                "label": label,
                "employee_id": None,
                "employee_name": None,
                "snapshot_path": None,
                "appearance_hash": signature,
            })

        self._associate_people(persons, face_detections)
        self._associate_objects(objects, persons, camera_id, frame, now)
        
        # Only return recognized people to the live feed to eliminate "blue boxes"
        recognized_persons = [p for p in persons if p.get("recognized")]
        return recognized_persons, objects

    def _associate_people(self, persons: list[dict], face_detections: list[dict]):
        for person in persons:
            best_face = None
            best_score = 0.0
            for face in face_detections:
                if not face.get("recognized"):
                    continue
                score = self._overlap_score(person["bbox"], face.get("bbox") or [])
                if score > best_score:
                    best_score = score
                    best_face = face

            if best_face is not None and best_score >= 0.08:
                person["employee_id"] = best_face.get("employee_id")
                person["employee_name"] = best_face.get("name")
                person["name"] = best_face.get("name")
                person["face_id"] = best_face.get("face_id")
                person["recognized"] = True

    def _associate_objects(self, objects: list[dict], persons: list[dict], camera_id: str, frame: np.ndarray, now: float):
        for obj in objects:
            matched_person = self._match_person(obj["bbox"], persons)
            if matched_person is not None:
                obj["employee_id"] = matched_person.get("employee_id")
                obj["employee_name"] = matched_person.get("employee_name")

            object_id = self._resolve_object_id(
                camera_id=None,
                object_label=obj["label"],
                appearance_hash=obj.get("appearance_hash"),
                track_id=obj.get("track_id"),
                employee_id=obj.get("employee_id"),
            )
            obj["object_id"] = object_id
            obj["snapshot_path"] = self._save_snapshot(frame, object_id, obj["bbox"])

            existing = self._db.get_object_detection(object_id)
            first_seen = existing["first_seen"] if existing else now
            occurrence_count = (existing["occurrence_count"] if existing and existing.get("occurrence_count") else 0) + 1
            self._db.upsert_object_detection({
                "object_id": object_id,
                "camera_id": camera_id,
                "employee_id": obj.get("employee_id"),
                "employee_name": obj.get("employee_name"),
                "object_label": obj["label"],
                "confidence": obj.get("confidence"),
                "track_id": obj.get("track_id"),
                "snapshot_path": obj["snapshot_path"],
                "bbox": obj.get("bbox"),
                "appearance_hash": obj.get("appearance_hash"),
                "first_seen": first_seen,
                "last_seen": now,
                "occurrence_count": occurrence_count,
            })

    def _resolve_object_id(self, camera_id: Optional[str], object_label: str, appearance_hash: Optional[str], track_id: Optional[str], employee_id: Optional[str]) -> str:
        cache_key = f"{camera_id}:{track_id}" if camera_id is not None and track_id is not None else track_id
        if cache_key and cache_key in self._track_to_object_id:
            return self._track_to_object_id[cache_key]

        match = None
        if appearance_hash:
            match = self._db.find_recent_object_match(
                object_label=object_label,
                appearance_hash=appearance_hash,
                camera_id=None,
                employee_id=employee_id,
            )

        if match:
            object_id = match["object_id"]
        else:
            # 2. Check local memory cache with Hamming Distance for flexibility
            object_id = None
            if appearance_hash:
                best_sig_match = None
                min_dist = 6  # Allow up to 5 bits difference in 64-bit hash (approx 92% similarity)
                
                for sig, cached_id in self._signature_to_object_id.items():
                    dist = self._hamming_dist(appearance_hash, sig)
                    if dist < min_dist:
                        min_dist = dist
                        best_sig_match = cached_id
                
                if best_sig_match:
                    object_id = best_sig_match
            
            if not object_id:
                slug = object_label.replace(" ", "-")[:18]
                object_id = f"OBJ-{slug}-{uuid.uuid4().hex[:8]}"

        if cache_key:
            self._track_to_object_id[cache_key] = object_id
        if appearance_hash and track_id is not None:
            self._track_to_signature[cache_key] = appearance_hash
            self._track_last_seen[cache_key] = time.time()
            self._signature_to_object_id[appearance_hash] = object_id
            self._signature_last_seen[appearance_hash] = time.time()
        return object_id

    def _match_person(self, bbox: list[int], persons: list[dict]) -> Optional[dict]:
        best_person = None
        best_score = 0.0
        for person in persons:
            score = self._overlap_score(bbox, person["bbox"])
            if score > best_score:
                best_score = score
                best_person = person
        if best_person is not None and best_score >= 0.12:
            return best_person
        return None

    def draw_detections(self, frame: np.ndarray, persons: list[dict], objects: list[dict]) -> np.ndarray:
        for person in persons:
            if not person.get("recognized"):
                continue
            x1, y1, x2, y2 = person["bbox"]
            color = (0, 220, 100)
            label = person.get("employee_name") or "Unknown person"
            if person.get("face_id"):
                label = f"{label} [{person['face_id']}]"
            self._draw_box(frame, x1, y1, x2, y2, label, color)

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            color = (0, 170, 255)
            label = f"{obj['label']} [{obj['object_id']}]"
            if obj.get("employee_name"):
                label = f"{label} -> {obj['employee_name']}"
            self._draw_box(frame, x1, y1, x2, y2, label, color)
        return frame

    def _draw_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color: tuple[int, int, int]):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(frame, (x1, label_y - th - baseline - 4), (x1 + tw + 6, label_y + 2), color, -1)
        cv2.putText(frame, label, (x1 + 3, label_y - baseline), font, font_scale, (255, 255, 255), thickness)

    def _save_snapshot(self, frame: np.ndarray, object_id: str, bbox: list[int]) -> str:
        x1, y1, x2, y2 = bbox
        crop = self._safe_crop(frame, bbox)
        object_dir = OBJECT_UPLOAD_DIR / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = object_dir / "snapshot.jpg"
        if crop.size:
            cv2.imwrite(str(snapshot_path), crop)
        return str(snapshot_path.relative_to(BASE_DIR / "uploads"))

    def _safe_crop(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2].copy()

    def _appearance_hash(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return "0"
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
        mean = float(small.mean())
        bits = (small > mean).astype(np.uint8).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return f"{value:016x}"

    def _hamming_dist(self, s1: str, s2: str) -> int:
        """Calculate Hamming distance between two hex strings."""
        try:
            v1 = int(s1, 16)
            v2 = int(s2, 16)
            return bin(v1 ^ v2).count('1')
        except Exception:
            return 999

    def _overlap_score(self, a: list[int], b: list[int]) -> float:
        if not a or not b:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _synthetic_track_id(self, label: str, bbox: list[int], signature: Optional[str]) -> str:
        x1, y1, x2, y2 = bbox
        bucket_x = int((x1 + x2) / 2 / 40)
        bucket_y = int((y1 + y2) / 2 / 40)
        sig_part = signature[:10] if signature else "nosig"
        return f"{label}:{bucket_x}:{bucket_y}:{sig_part}"
