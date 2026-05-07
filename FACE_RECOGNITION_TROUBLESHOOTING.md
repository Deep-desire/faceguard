# Face Recognition Troubleshooting Guide

## Problem: Registered Faces Not Detected in Live Feed

If registered employees' faces are not appearing in the live feed, follow this diagnostic process:

---

## Step 1: Check System Health

**Endpoint**: `GET http://localhost:8000/api/health`

```bash
curl http://localhost:8000/api/health
```

**Expected output** (example):
```json
{
  "status": "ok",
  "active_cameras": 1,
  "registered_faces": 3,
  "embed_backend": "insightface_onnx",
  "total_employees": 3,
  "insight_onnx": "available",
  "deepface": "available",
  "yolo": "available"
}
```

**What to check:**
- ✅ `registered_faces` should be > 0 (employees registered)
- ✅ `embed_backend` should be `insightface_onnx` (best), `deepface`, or `insightface`
- ✅ At least one backend should be "available"
- ⚠️  If `registered_faces: 0`, no employees are loaded

---

## Step 2: Verify Embeddings Are Loaded

**Check the backend logs** when the server starts (look for these messages):

```
✅ Loaded 3 employee embeddings from database
✅ Registered face: John Doe (EMP-001) — emb norm=0.999
✅ Registered face: Jane Smith (EMP-002) — emb norm=1.000
...
```

**If you don't see these messages:**
- Employees might not be registered in the database
- The database might not be initialized
- Solution: Register employees through the frontend

---

## Step 3: Check Backend Logs for Embedding Extraction

When registering a new employee, watch the backend logs for:

```
✅ Extracted embedding #1 for John Doe
✅ Extracted embedding #2 for John Doe
✅ Employee registered: John Doe | EMP-001 | 2 photos averaged
✅ Registered face: John Doe (EMP-001) — emb norm=1.000
```

**If you see warnings:**
```
⚠️  No embedding extracted from image #1 for John Doe
```

**Causes & Solutions:**
- Face not detected in the photo → Upload clearer face photos
- Image is corrupted → Try a different image format (JPG, PNG)
- Embedding backend failed → Check which backend is active

---

## Step 4: Monitor Live Feed Matching

**Enable DEBUG logging** to see cosine similarity scores during live detection.

Edit `backend/face_engine.py` line 8:
```python
DEBUG_MATCHING = True  # Already enabled by default
```

When faces are detected in live feed, you'll see debug logs:
```
Cosine match: John Doe = 0.8234 (✅ MATCHED, threshold=0.40)
Cosine match: Unknown Person = 0.3100 (⚠️  BELOW_THRESHOLD, threshold=0.40)
```

**What this tells you:**
- ✅ `MATCHED`: Face was recognized and will appear in live feed
- ⚠️  `BELOW_THRESHOLD`: Face detected but similarity score too low
- No log entry: Face not detected in frame

---

## Step 5: Common Issues & Solutions

### Issue 1: "Registered Faces: 0" in Health Check

**Cause**: Employees are in database but not loaded into memory

**Solution**: 
1. Restart the backend server
2. Verify employees are saved in the database:
   ```bash
   # On Windows, from backend folder:
   python -c "from database import Database; db = Database(); print([e['name'] for e in db.get_all_employees()])"
   ```

---

### Issue 2: Cosine Similarity Too Low (Below Threshold)

**Example**: `Cosine match: John Doe = 0.35 (⚠️  BELOW_THRESHOLD, threshold=0.40)`

**Causes:**
- Face photos during registration were different from live camera angle
- Camera lens distortion or RTSP compression causing face misalignment
- Embedding backend different between registration and live detection

**Solutions:**
- Re-register employee with photos from similar angles as the camera
- Upload photos with better lighting and face clarity
- Try different backend (if available):
  - Best: InsightFace ONNX (uses landmark-based alignment)
  - Fallback: DeepFace (uses TensorFlow)

---

### Issue 3: "No Embedding Extracted from Image"

**Cause**: Face detection failed in the registered photo

**Solution:**
- Re-upload photos where the face is clearly visible
- Ensure face is well-lit and frontal
- Try JPG format instead of PNG/WEBP

---

### Issue 4: Backend Shows "insightface_onnx: unavailable"

**Cause**: ONNX models not found

**Solution:**
1. Check if `backend/insightface_models/buffalo_l/` contains:
   - `w600k_r50.onnx` (embeddings)
   - `det_10g.onnx` (detection)
2. If missing, download from: https://github.com/deepinsight/insightface/releases
3. Place in correct folder structure

---

## Step 6: Optimize Thresholds

If faces are not matching at threshold 0.40, you can lower it:

**File**: `backend/face_engine.py`, line ~7

```python
COSINE_THRESHOLD = 0.40   # Try lowering to 0.35 if matches fail
```

**Trade-offs:**
- Lower threshold = more matches but more false positives
- Higher threshold = fewer false positives but may miss valid faces

**Recommended range**: 0.35 - 0.45

---

## Step 7: Test End-to-End

1. **Register an employee**:
   - Upload 3-5 clear face photos from different angles
   - Watch backend logs for embedding extraction

2. **Check health**:
   ```bash
   curl http://localhost:8000/api/health
   ```
   Should show `registered_faces: 1+`

3. **View live feed**:
   - Open LiveFeed tab in frontend
   - Look for employee names with green bounding boxes
   - Watch backend console for matching scores

4. **Check events**:
   - Go to Events tab
   - Filter by "Recognized Only"
   - Should show detections of registered employees

---

## Advanced Debugging

### Check Actual Embedding Values

```python
from database import Database
from face_engine import FaceEngine

db = Database()
fe = FaceEngine()
fe.initialize()
fe.load_from_db(db)

# See what embeddings are loaded
for emp_id, data in fe._registry.items():
    print(f"{data['name']}: norm={np.linalg.norm(data['embedding']):.4f}")
```

### Compare Registration vs Live Embeddings

If you suspect embeddings are mismatched:
1. Extract embedding from saved registration image
2. Extract embedding from live camera frame
3. Compute cosine similarity manually

```python
import numpy as np

reg_emb = np.array([...])  # from registration
live_emb = np.array([...]) # from live frame

similarity = np.dot(reg_emb, live_emb)
print(f"Similarity: {similarity:.4f}")
```

---

## Contact Support

If face recognition still doesn't work after these steps:

1. **Collect diagnostics**:
   - Backend startup logs (including health check output)
   - Employee registration logs
   - Live feed matching logs

2. **Check limitations**:
   - RTSP stream quality (artifacts reduce face alignment)
   - CCTV camera angle (profile faces harder to match than frontal)
   - Lighting conditions (low-light reduces recognition)

3. **Verify setup**:
   - Are embeddings being stored in database?
   - Is backend initialized before startup?
   - Are photos actually being saved?
