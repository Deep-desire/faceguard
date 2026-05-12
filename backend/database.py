"""
FaceGuard Pro - Database Layer (SQLite via sqlite3)
Handles cameras, employees, detection events, and object detections.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "faceguard.db"


class Database:
    def __init__(self):
        self._path = str(DB_PATH)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    rtsp_url    TEXT NOT NULL,
                    location    TEXT,
                    is_active   INTEGER DEFAULT 1,
                    created_at  REAL
                );

                CREATE TABLE IF NOT EXISTS employees (
                    id              TEXT PRIMARY KEY,
                    face_id         TEXT UNIQUE NOT NULL,
                    name            TEXT NOT NULL,
                    department      TEXT,
                    employee_code   TEXT UNIQUE NOT NULL,
                    email           TEXT,
                    phone           TEXT,
                    image_paths     TEXT DEFAULT '[]',
                    embedding       TEXT,
                    created_at      REAL
                );

                CREATE TABLE IF NOT EXISTS detection_events (
                    id              TEXT PRIMARY KEY,
                    camera_id       TEXT NOT NULL,
                    employee_id     TEXT,
                    face_id         TEXT,
                    employee_name   TEXT,
                    confidence      REAL,
                    track_id        TEXT,
                    timestamp       REAL,
                    frame_b64       TEXT,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_ts   ON detection_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_cam  ON detection_events(camera_id);
                CREATE INDEX IF NOT EXISTS idx_events_face ON detection_events(face_id);

                CREATE TABLE IF NOT EXISTS object_detections (
                    object_id        TEXT PRIMARY KEY,
                    camera_id        TEXT NOT NULL,
                    employee_id      TEXT,
                    employee_name    TEXT,
                    object_label     TEXT NOT NULL,
                    confidence       REAL,
                    track_id         TEXT,
                    snapshot_path    TEXT,
                    bbox             TEXT,
                    appearance_hash  TEXT,
                    first_seen       REAL,
                    last_seen        REAL,
                    occurrence_count INTEGER DEFAULT 1,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id)
                );

                CREATE INDEX IF NOT EXISTS idx_objects_last_seen ON object_detections(last_seen);
                CREATE INDEX IF NOT EXISTS idx_objects_cam       ON object_detections(camera_id);
                CREATE INDEX IF NOT EXISTS idx_objects_employee  ON object_detections(employee_id);
            """)

            self._ensure_column("object_detections", "appearance_hash", "TEXT")

    # ── Cameras ────────────────────────────────────────────────────────────────

    def get_all_cameras(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM cameras ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_camera(self, cam_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
            return dict(row) if row else None

    def insert_camera(self, cam: dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cameras (id,name,rtsp_url,location,is_active,created_at) VALUES (?,?,?,?,?,?)",
                (cam["id"], cam["name"], cam["rtsp_url"], cam.get("location"),
                 int(cam.get("is_active", True)), cam["created_at"]),
            )

    def update_camera(self, cam_id: str, updates: dict):
        if not updates:
            return
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [cam_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE cameras SET {cols} WHERE id=?", vals)

    def delete_camera(self, cam_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM cameras WHERE id=?", (cam_id,))

    # ── Employees ──────────────────────────────────────────────────────────────

    def get_all_employees(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM employees ORDER BY created_at DESC").fetchall()
            return [self._fmt_emp(dict(r)) for r in rows]

    def get_employee(self, emp_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM employees WHERE id=?", (emp_id,)).fetchone()
            return dict(row) if row else None

    def get_all_embeddings(self) -> list[dict]:
        """Return id, face_id, name, embedding for face engine reload."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, face_id, name, embedding FROM employees WHERE embedding IS NOT NULL"
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_employee(self, emp: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO employees
                   (id,face_id,name,department,employee_code,email,phone,image_paths,embedding,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (emp["id"], emp["face_id"], emp["name"], emp.get("department"),
                 emp["employee_code"], emp.get("email"), emp.get("phone"),
                 emp.get("image_paths", "[]"), emp.get("embedding"), emp["created_at"]),
            )

    def update_employee(self, emp_id: str, updates: dict):
        if not updates:
            return
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [emp_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE employees SET {cols} WHERE id=?", vals)

    def delete_employee(self, emp_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM employees WHERE id=?", (emp_id,))

    # ── Detection Events ───────────────────────────────────────────────────────

    def insert_event(self, event: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO detection_events
                   (id,camera_id,employee_id,face_id,employee_name,confidence,track_id,timestamp,frame_b64)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (event["id"], event["camera_id"], event.get("employee_id"),
                 event.get("face_id"), event.get("employee_name"),
                 event.get("confidence"), event.get("track_id"),
                 event["timestamp"], event.get("frame_b64")),
            )

    def get_events(self, limit: int = 100, camera_id: Optional[str] = None) -> list[dict]:
        with self._conn() as conn:
            if camera_id:
                rows = conn.execute(
                    "SELECT * FROM detection_events WHERE camera_id=? ORDER BY timestamp DESC LIMIT ?",
                    (camera_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM detection_events ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_events_today(self) -> int:
        start = time.time() - 86400
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM detection_events WHERE timestamp >= ?", (start,)
            ).fetchone()
            return row[0]

    def get_unique_detections_today(self) -> int:
        start = time.time() - 86400
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT face_id) FROM detection_events WHERE timestamp >= ? AND face_id IS NOT NULL",
                (start,),
            ).fetchone()
            return row[0]

    # ── Object Detections ─────────────────────────────────────────────────────

    def insert_object_detection(self, obj: dict):
        self.upsert_object_detection(obj)

    def upsert_object_detection(self, obj: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO object_detections
                   (object_id,camera_id,employee_id,employee_name,object_label,confidence,track_id,snapshot_path,bbox,appearance_hash,first_seen,last_seen,occurrence_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(object_id) DO UPDATE SET
                       camera_id=excluded.camera_id,
                       employee_id=excluded.employee_id,
                       employee_name=excluded.employee_name,
                       object_label=excluded.object_label,
                       confidence=excluded.confidence,
                       track_id=excluded.track_id,
                       snapshot_path=excluded.snapshot_path,
                       bbox=excluded.bbox,
                       appearance_hash=excluded.appearance_hash,
                       first_seen=MIN(object_detections.first_seen, excluded.first_seen),
                       last_seen=excluded.last_seen,
                       occurrence_count=excluded.occurrence_count""",
                (
                    obj["object_id"],
                    obj["camera_id"],
                    obj.get("employee_id"),
                    obj.get("employee_name"),
                    obj["object_label"],
                    obj.get("confidence"),
                    obj.get("track_id"),
                    obj.get("snapshot_path"),
                    json.dumps(obj.get("bbox")) if obj.get("bbox") is not None else None,
                    obj.get("appearance_hash"),
                    obj.get("first_seen"),
                    obj.get("last_seen"),
                    obj.get("occurrence_count", 1),
                ),
            )

    def get_object_detection(self, object_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM object_detections WHERE object_id=?", (object_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            data["bbox"] = json.loads(data["bbox"]) if data.get("bbox") else None
            return data

    def find_recent_object_match(
        self,
        object_label: str,
        appearance_hash: str,
        camera_id: Optional[str] = None,
        employee_id: Optional[str] = None,
        max_age_seconds: int = 900,
        max_hamming_distance: int = 8,
    ) -> Optional[dict]:
        cutoff = time.time() - max_age_seconds
        query = "SELECT * FROM object_detections WHERE object_label=? AND last_seen >= ?"
        params: list = [object_label, cutoff]

        if camera_id:
            query += " AND camera_id=?"
            params.append(camera_id)
        if employee_id:
            query += " AND employee_id=?"
            params.append(employee_id)

        query += " ORDER BY last_seen DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            for row in rows:
                candidate = dict(row)
                candidate_hash = candidate.get("appearance_hash")
                if candidate_hash and self._hash_distance(candidate_hash, appearance_hash) <= max_hamming_distance:
                    candidate["bbox"] = json.loads(candidate["bbox"]) if candidate.get("bbox") else None
                    return candidate
        return None

    def get_object_detections(
        self,
        limit: int = 100,
        camera_id: Optional[str] = None,
        employee_id: Optional[str] = None,
    ) -> list[dict]:
        query = "SELECT * FROM object_detections"
        params: list = []
        clauses = []

        if camera_id:
            clauses.append("camera_id=?")
            params.append(camera_id)
        if employee_id:
            clauses.append("employee_id=?")
            params.append(employee_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            detections = [dict(r) for r in rows]
            for detection in detections:
                detection["bbox"] = json.loads(detection["bbox"]) if detection.get("bbox") else None
            return detections

    def get_objects_today(self) -> int:
        start = time.time() - 86400
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM object_detections WHERE last_seen >= ?",
                (start,),
            ).fetchone()
            return row[0]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _fmt_emp(self, emp: dict) -> dict:
        emp["image_paths"] = json.loads(emp.get("image_paths") or "[]")
        emp.pop("embedding", None)
        return emp

    def _ensure_column(self, table: str, column: str, definition: str):
        with self._conn() as conn:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _hash_distance(self, hash_a: str, hash_b: str) -> int:
        try:
            return int(bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1"))
        except Exception:
            return 999
