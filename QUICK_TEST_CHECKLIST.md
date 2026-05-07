# FaceGuard Pro - Quick Testing Checklist

## Pre-Test Verification

- [ ] Backend running: `cd backend && .\start.ps1`
- [ ] Frontend running on `localhost:3000`
- [ ] RTSP cameras configured and accessible

---

## 1️⃣ System Health Check (< 1 min)

```bash
curl http://localhost:8000/api/health
```

**Expected response:**
```json
{
  "status": "ok",
  "active_cameras": 1,
  "registered_faces": 0,
  "embed_backend": "insightface_onnx",
  "total_employees": 0,
  "insight_onnx": "available",
  "deepface": "available",
  "yolo": "available"
}
```

**✅ Success indicators:**
- At least 2 backends "available"
- "registered_faces" increases as you add employees
- "embed_backend" is one of: insightface_onnx, deepface, insightface

---

## 2️⃣ Register Test Employee (2-3 min)

1. Go to **Employees** tab in frontend
2. Click **Add Employee**
3. Fill in:
   - Name: `John Test`
   - Department: `IT`
   - Employee Code: `TEST001`
   - Upload 3-5 clear face photos (different angles)
4. Click **Register**

**Watch backend logs for:**
```
✅ Extracted embedding #1 for John Test
✅ Extracted embedding #2 for John Test
✅ Extracted embedding #3 for John Test
✅ Employee registered: John Test | EMP-TEST001 | 3 photos averaged
✅ Registered face: John Test (EMP-TEST001) — emb norm=0.999
```

**Check health again:**
```bash
curl http://localhost:8000/api/health
```

Should show: `"registered_faces": 1`

---

## 3️⃣ Test Live Feed Detection (5 min)

1. Go to **Cameras** tab → select a camera
2. View **Live Feed**
3. Show the camera John Test's face
4. Watch for:
   - ✅ Green bounding box appears
   - ✅ Name label shows "John Test [EMP-TEST001]"
   - ✅ Backend logs show:
     ```
     Cosine match: John Test = 0.85 (✅ MATCHED, threshold=0.40)
     ```

**Troubleshoot if no detection:**
- Ensure face is well-lit and frontal
- Try getting closer to camera
- Check camera angle matches registration photos
- Watch logs for "⚠️ BELOW_THRESHOLD" messages

---

## 4️⃣ Verify Events Recording (2 min)

1. Go to **Events** tab
2. Filter by camera (if multi-camera)
3. Should see entries for John Test:
   ```
   John Test [EMP-TEST001] — RECOGNIZED
   ```
4. Click entry to see details:
   - Timestamp
   - Camera
   - Bounding box screenshot
   - Confidence score

---

## 5️⃣ Test Multiple Employees (5-10 min)

1. Register 2-3 more test employees
2. Verify all appear in live feed
3. Check that each has unique name label
4. Verify no false positives (unknown faces stay as "Unknown")

**Expected behavior:**
- Registered faces → green box with name
- Unknown faces → no box or no name label
- Multiple employees visible simultaneously

---

## 🔧 Debugging Commands

**View all employees in database:**
```bash
cd backend
python -c "from database import Database; db = Database(); print('\n'.join([f\"{e['name']} ({e['face_id']})\" for e in db.get_all_employees()]))"
```

**Check database file:**
```bash
ls -la backend/faceguard.db
# Should be > 1 MB
```

**Monitor backend logs in real-time:**
```bash
# In PowerShell
cd backend
.\start.ps1 2>&1 | Tee-Object -FilePath logs.txt
```

**Check frontend console for errors:**
- Right-click browser → Inspect → Console tab
- Should be empty of errors

---

## 📊 Performance Expectations

| Metric | Target | Actual |
|--------|--------|--------|
| Face detection latency | < 200ms | __ |
| Embedding extraction | < 100ms | __ |
| Cosine match time | < 10ms | __ |
| WebSocket frame rate | 15 FPS | __ |
| Memory usage (backend) | < 2GB | __ |

---

## ❌ Common Issues

| Issue | Solution |
|-------|----------|
| **No faces detected** | Check face quality, lighting, angle |
| **"⚠️ BELOW_THRESHOLD" in logs** | Re-register with clearer photos |
| **No backend logs** | Restart backend with `.\start.ps1` |
| **WebSocket connection drops** | Check browser console for errors |
| **Only first employee recognized** | Verify all embeddings loaded (health check) |

---

## ✅ Test Pass Criteria

**All of the following must pass:**

- [ ] Health endpoint returns "ok" status
- [ ] Employee registration shows embedding extraction logs
- [ ] Registered face appears in live feed with name
- [ ] Backend logs show cosine match score ≥ 0.40
- [ ] Events tab records recognized detections
- [ ] Multiple employees work simultaneously
- [ ] Unknown faces don't show as recognized
- [ ] Confidence scores appear in events
- [ ] No WebSocket disconnections during 5-min test
- [ ] Backend memory stays under 2GB

---

## 📝 Test Results

Date: ____________
Tester: ____________

| Test | Pass | Fail | Notes |
|------|------|------|-------|
| Health endpoint | ☐ | ☐ | |
| Employee registration | ☐ | ☐ | |
| Face embedding extraction | ☐ | ☐ | |
| Live feed detection | ☐ | ☐ | |
| Events recording | ☐ | ☐ | |
| Multiple employees | ☐ | ☐ | |
| No false positives | ☐ | ☐ | |
| Performance | ☐ | ☐ | |

**Overall Result:** ☐ PASS ☐ FAIL

**Issues Found:**
- 
- 
- 

---

## 🚀 Next Steps

After successful testing:

1. Test with real RTSP cameras
2. Configure automated event alerting
3. Set up database backups
4. Deploy to production hardware
5. Configure RBAC for multi-user access
