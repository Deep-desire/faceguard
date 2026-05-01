"""
FaceGuard Pro - Database Layer (SQLite via sqlite3)
Handles cameras, employees, and detection events.
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
            """)

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

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _fmt_emp(self, emp: dict) -> dict:
        emp["image_paths"] = json.loads(emp.get("image_paths") or "[]")
        emp.pop("embedding", None)
        return emp
