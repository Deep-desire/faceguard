"""
insightface_onnx.py  –  InsightFace buffalo_l via pure onnxruntime
==================================================================
Models used (in insightface_models/buffalo_l/):
  det_10g.onnx   – SCRFD-10GF face detector  (outputs: 9 arrays, 3 strides × score/bbox/kps)
  w600k_r50.onnx – ArcFace R50 recogniser    (512-D embeddings)

Key fix vs. previous version:
  • SCRFD has 9 separate outputs (3 strides × score, bbox, kps) – not 2 concatenated arrays.
  • ArcFace requires a geometrically-aligned 112×112 face (using 5 landmarks).
    Plain resize gives ~0 cosine similarity between registration & live crops.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger("faceguard.insightface_onnx")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
MODEL_DIR = _HERE / "insightface_models" / "buffalo_l"
DET_MODEL = MODEL_DIR / "det_10g.onnx"
REC_MODEL = MODEL_DIR / "w600k_r50.onnx"

# ── Detection ──────────────────────────────────────────────────────────────────
DET_SIZE    = (640, 640)       # model input resolution
STRIDES     = [8, 16, 32]
NUM_ANCHORS = 2
CONF_THRESH = 0.35
NMS_THRESH  = 0.45

# ── ArcFace alignment – canonical 112×112 landmark template ───────────────────
ARCFACE_DST = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


class InsightFaceONNX:
    """
    Drop-in for insightface.app.FaceAnalysis – no C++ build tools needed.

    Public API
    ----------
    get_faces(img_bgr)         → list of face dicts with bbox, kps, embedding
    get_embedding(face_crop)   → 512-D embedding from an already-cropped face
    detect_and_embed(img_bgr)  → embedding of largest detected face (for registration)
    """

    def __init__(self):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]

        log.info("Loading InsightFace ONNX models…")
        self._det = ort.InferenceSession(str(DET_MODEL), sess_options=opts, providers=providers)
        self._rec = ort.InferenceSession(str(REC_MODEL), sess_options=opts, providers=providers)

        self._det_in = self._det.get_inputs()[0].name
        self._rec_in = self._rec.get_inputs()[0].name

        # Inspect output count so we can handle both 9-output and 6-output variants
        self._det_out_names = [o.name for o in self._det.get_outputs()]
        n = len(self._det_out_names)
        self._has_kps = (n == 9)   # 9 = score+bbox+kps per stride; 6 = score+bbox only
        log.info(f"det_10g outputs: {n}  (landmarks={'yes' if self._has_kps else 'no'})")

        # Pre-build anchor centres [total_anchors, 2]
        self._anchor_centers, self._anchor_strides = self._build_anchors(DET_SIZE)
        log.info("InsightFace ONNX ready ✅")

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_faces(self, img_bgr: np.ndarray) -> List[dict]:
        """
        Full pipeline on a BGR frame.
        Returns list of:
          { bbox: [x1,y1,x2,y2], conf: float, kps: np.ndarray(5,2)|None,
            embedding: np.ndarray(512,) }
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        h, w = img_bgr.shape[:2]
        resized, scale, (px, py) = self._letterbox(img_bgr, DET_SIZE)
        blob = self._preprocess_det(resized)
        raw  = self._det.run(None, {self._det_in: blob})

        boxes, scores, kps_list = self._decode(raw, DET_SIZE)
        if len(boxes) == 0:
            return []

        faces = []
        for box, score, kps in zip(boxes, scores, kps_list):
            # Map back to original image coords
            x1 = max(0, int((box[0] - px) / scale))
            y1 = max(0, int((box[1] - py) / scale))
            x2 = min(w,  int((box[2] - px) / scale))
            y2 = min(h,  int((box[3] - py) / scale))

            orig_kps = None
            if kps is not None:
                orig_kps = kps.copy()
                orig_kps[:, 0] = (kps[:, 0] - px) / scale
                orig_kps[:, 1] = (kps[:, 1] - py) / scale

            # Align & embed
            aligned = self._align(img_bgr, x1, y1, x2, y2, orig_kps)
            emb      = self._embed(aligned)

            faces.append({
                "bbox":      [x1, y1, x2, y2],
                "conf":      float(score),
                "kps":       orig_kps,
                "embedding": emb,
            })

        return faces

    def detect_and_embed(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return embedding of the largest face found in the image. Used at registration."""
        faces = self.get_faces(img_bgr)
        if not faces:
            return None
        # pick largest by bbox area
        largest = max(faces, key=lambda f: (f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1]))
        return largest["embedding"]

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Embed an already-cropped face region (no detection).
        Crops are assumed to be reasonably well-framed; we just resize+embed.
        """
        if face_crop is None or face_crop.size == 0:
            return None
        return self._embed(face_crop)

    # ── Detection internals ─────────────────────────────────────────────────────

    def _letterbox(self, img, target):
        th, tw = target
        h, w   = img.shape[:2]
        scale  = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((th, tw, 3), 114, dtype=np.uint8)
        px = (tw - nw) // 2
        py = (th - nh) // 2
        canvas[py:py+nh, px:px+nw] = resized
        return canvas, scale, (px, py)

    def _preprocess_det(self, img):
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        return img.transpose(2, 0, 1)[np.newaxis]   # NCHW

    def _build_anchors(self, det_size):
        """Build anchor centre grid and per-anchor stride array."""
        centers, strides = [], []
        h, w = det_size
        for stride in STRIDES:
            gh, gw = h // stride, w // stride
            for gy in range(gh):
                for gx in range(gw):
                    for _ in range(NUM_ANCHORS):
                        centers.append([(gx + 0.5) * stride, (gy + 0.5) * stride])
                        strides.append(stride)
        return (np.array(centers, dtype=np.float32),
                np.array(strides, dtype=np.float32))

    def _decode(self, outputs, det_size) -> Tuple[list, list, list]:
        """
        SCRFD output layout (9 outputs with kps, 6 without):
          outputs[0..2]  = scores per stride  (N_s, 1)
          outputs[3..5]  = bboxes per stride  (N_s, 4)
          outputs[6..8]  = kps    per stride  (N_s, 10)   ← only if has_kps
        """
        n_strides = len(STRIDES)

        score_list, bbox_list, kps_list = [], [], []
        offset = 0
        h, w = det_size

        for i, stride in enumerate(STRIDES):
            gh, gw = h // stride, w // stride
            n = gh * gw * NUM_ANCHORS

            scores = outputs[i].reshape(-1)               # (N_s,)
            bboxes = outputs[i + n_strides].reshape(-1, 4) # (N_s, 4)
            if self._has_kps:
                kps = outputs[i + 2 * n_strides].reshape(-1, 5, 2)  # (N_s, 5, 2)
            else:
                kps = np.zeros((n, 5, 2), dtype=np.float32)

            ac  = self._anchor_centers[offset:offset+n]   # (N_s, 2)
            str_= self._anchor_strides[offset:offset+n]   # (N_s,)
            offset += n

            # Decode boxes (ltrb distance format)
            x1 = ac[:, 0] - bboxes[:, 0] * str_
            y1 = ac[:, 1] - bboxes[:, 1] * str_
            x2 = ac[:, 0] + bboxes[:, 2] * str_
            y2 = ac[:, 1] + bboxes[:, 3] * str_
            decoded_boxes = np.stack([x1, y1, x2, y2], axis=1)

            # Decode keypoints
            decoded_kps = ac[:, np.newaxis, :] + kps * str_[:, np.newaxis, np.newaxis]

            score_list.append(scores)
            bbox_list.append(decoded_boxes)
            kps_list.append(decoded_kps)

        all_scores = np.concatenate(score_list)
        all_boxes  = np.concatenate(bbox_list)
        all_kps    = np.concatenate(kps_list)

        keep = all_scores >= CONF_THRESH
        all_scores = all_scores[keep]
        all_boxes  = all_boxes[keep]
        all_kps    = all_kps[keep]

        if len(all_scores) == 0:
            return [], [], []

        nms_keep = self._nms(all_boxes, all_scores)
        return (all_boxes[nms_keep].tolist(),
                all_scores[nms_keep].tolist(),
                [all_kps[i] for i in nms_keep])

    def _nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas  = (x2-x1) * (y2-y1)
        order  = scores.argsort()[::-1]
        keep   = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            iou = (np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)) / \
                  (areas[i] + areas[order[1:]] - (np.maximum(0, xx2-xx1)*np.maximum(0, yy2-yy1)) + 1e-6)
            order = order[1:][iou < NMS_THRESH]
        return keep

    # ── Face alignment & embedding ──────────────────────────────────────────────

    def _align(self, img, x1, y1, x2, y2, kps5=None):
        """Produce a 112×112 geometrically-aligned face for ArcFace."""
        if kps5 is not None:
            M, _ = cv2.estimateAffinePartial2D(
                kps5.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS
            )
            if M is not None:
                return cv2.warpAffine(img, M, (112, 112), borderValue=0)
        # Fallback: simple crop + resize
        crop = img[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _embed(self, face_112: np.ndarray) -> np.ndarray:
        """ArcFace forward pass on a 112×112 BGR crop."""
        img = cv2.resize(face_112, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        blob = img.transpose(2, 0, 1)[np.newaxis]
        out  = self._rec.run(None, {self._rec_in: blob})[0][0].astype(np.float32)
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out
