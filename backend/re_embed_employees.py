"""
re_embed_employees.py
---------------------
Re-computes all employee face embeddings using the fixed InsightFace ONNX
pipeline (with proper 5-point face alignment).

Run once after updating insightface_onnx.py:
  cd backend
  venv\Scripts\activate
  python re_embed_employees.py
"""

import json, logging, os
import cv2, numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("re_embed")

from database import Database
from insightface_onnx import InsightFaceONNX, MODEL_DIR

if not (MODEL_DIR / "w600k_r50.onnx").exists():
    raise SystemExit("ONNX models not found – run from backend/ directory")

db   = Database(); db.init()
app  = InsightFaceONNX()

BASE_UPLOAD = os.path.join(os.path.dirname(__file__), "uploads")
employees   = db.get_all_employees()
log.info(f"Found {len(employees)} employees to re-embed")

success = fail = 0
for emp in employees:
    raw_paths = emp.get("image_paths", "[]")
    paths = raw_paths if isinstance(raw_paths, list) else json.loads(raw_paths)
    embeddings = []

    for rel_path in paths:
        img_path = os.path.join(BASE_UPLOAD, rel_path)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        emb = app.detect_and_embed(img)
        if emb is not None:
            embeddings.append(emb)
            log.info(f"  ✅ {emp['name']} – {rel_path}")
        else:
            log.warning(f"  ⚠️  No face found in {rel_path}")

    if not embeddings:
        log.error(f"  ❌ {emp['name']} – no valid embeddings, skipping")
        fail += 1
        continue

    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    db.update_employee(emp["id"], {"embedding": json.dumps(avg_emb.tolist())})
    log.info(f"  💾 {emp['name']} re-embedded ({len(embeddings)} photos)")
    success += 1

log.info(f"\nDone. {success} updated, {fail} failed.")
log.info("Restart the uvicorn server to load new embeddings.")
