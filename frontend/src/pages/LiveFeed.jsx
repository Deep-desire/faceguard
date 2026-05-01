import React, { useEffect, useState, useRef, useCallback } from "react";
import { getCameras, createCameraWS, createEventsWS } from "../api";
import { MonitorPlay, Wifi, WifiOff, Activity, User } from "lucide-react";
import { format, fromUnixTime } from "date-fns";

export default function LiveFeed() {
  const [cameras, setCameras]             = useState([]);
  const [frames, setFrames]               = useState({});
  const [cameraDetections, setCameraDetections] = useState({});
  const [events, setEvents]               = useState([]);
  const wsRefs      = useRef({});
  const reconnectTs = useRef({});   // cam_id → timestamp of last reconnect attempt
  const eventsWsRef = useRef(null);
  const mountedRef  = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    loadAndConnect();
    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, []);

  async function loadAndConnect() {
    const cams = await getCameras();
    if (!mountedRef.current) return;
    setCameras(cams);
    cams.filter(c => c.is_active).forEach(c => connectCamera(c));
    connectEventsWS();
  }

  function connectCamera(cam) {
    // Rate-limit reconnects: at most once per 3 s per camera
    const now = Date.now();
    if (reconnectTs.current[cam.id] && now - reconnectTs.current[cam.id] < 3000) return;
    reconnectTs.current[cam.id] = now;

    // Close stale socket if any
    const stale = wsRefs.current[cam.id];
    if (stale && stale.readyState < 2) {   // CONNECTING or OPEN
      stale.onclose = null;                // prevent its onclose from firing
      stale.close();
    }
    wsRefs.current[cam.id] = null;

    if (!mountedRef.current) return;

    const ws = createCameraWS(
      cam.id,
      (data) => {
        if (!mountedRef.current) return;
        if (data.type === "frame") {
          setFrames(f => ({ ...f, [cam.id]: data.frame }));
          setCameraDetections(d => ({ ...d, [cam.id]: data.detections || [] }));
        }
        // type === "status" → camera inactive, don't update frame
      },
      () => {
        // WebSocket closed — reconnect after 3 s with rate-limiting
        if (!mountedRef.current) return;
        setTimeout(() => {
          if (mountedRef.current) connectCamera(cam);
        }, 3000);
      }
    );
    wsRefs.current[cam.id] = ws;
  }

  function connectEventsWS() {
    eventsWsRef.current = createEventsWS((data) => {
      if (!mountedRef.current) return;
      if (data.type === "detection") {
        setEvents(prev => [data, ...prev].slice(0, 50));
      }
    });
  }

  function cleanup() {
    Object.values(wsRefs.current).forEach(ws => {
      if (ws) { ws.onclose = null; ws.close(); }
    });
    wsRefs.current = {};
    if (eventsWsRef.current) { eventsWsRef.current.onclose = null; eventsWsRef.current.close(); }
  }

  const activeCams = cameras.filter(c => c.is_active);

  return (
    <div className="page" style={{ paddingBottom: 0 }}>
      <div className="page-header">
        <div className="page-title">LIVE FEED</div>
        <div className="page-sub">Real-time CCTV face detection streams</div>
      </div>

      <div className="live-grid">
        {/* Camera Feeds */}
        <div className="live-feed-main">
          {activeCams.length === 0 ? (
            <div className="empty-state" style={{ gridColumn: "1/-1" }}>
              <MonitorPlay size={48} className="empty-state-icon" />
              <div className="empty-state-text">No active cameras. Enable cameras in the Cameras tab.</div>
            </div>
          ) : (
            activeCams.map(cam => (
              <CameraView
                key={cam.id}
                cam={cam}
                frame={frames[cam.id]}
                detections={cameraDetections[cam.id] || []}
              />
            ))
          )}
        </div>

        {/* Detection Event Panel */}
        <div className="event-panel">
          <div className="event-panel-header">
            <Activity size={16} style={{ color: "var(--accent)" }} />
            Detection Feed
            <span className="badge badge-green" style={{ marginLeft: "auto" }}>
              {events.length} events
            </span>
          </div>
          <div className="event-list">
            {events.length === 0 ? (
              <div className="empty-state" style={{ padding: "30px" }}>
                <Activity size={28} className="empty-state-icon" />
                <div className="empty-state-text">Awaiting detections...</div>
              </div>
            ) : (
              events.map((ev, i) => (
                <div className="event-item" key={i}>
                  {ev.face_image ? (
                    <img
                      src={`data:image/jpeg;base64,${ev.face_image}`}
                      alt=""
                      className="event-face-img"
                    />
                  ) : (
                    <div className="event-face-img" style={{
                      background: "var(--accent-dim)",
                      display: "flex", alignItems: "center", justifyContent: "center"
                    }}>
                      <User size={16} color="var(--accent)" />
                    </div>
                  )}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="event-name">{ev.employee_name || "Unknown"}</div>
                    <div className="event-detail">
                      {ev.face_id && <span style={{ color: "var(--accent)" }}>{ev.face_id}</span>}
                      {ev.confidence && <span> · {(ev.confidence * 100).toFixed(0)}%</span>}
                    </div>
                    <div className="event-detail" style={{ marginTop: 2 }}>
                      {ev.timestamp && format(fromUnixTime(ev.timestamp), "HH:mm:ss")}
                    </div>
                  </div>
                  <span className="badge badge-green" style={{ fontSize: 10 }}>✓</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function CameraView({ cam, frame, detections }) {
  const imgRef    = useRef(null);
  const [imgSize, setImgSize] = useState({ w: 1, h: 1 }); // natural size of streamed frame
  const online = !!frame;

  // Capture actual rendered image dimensions for correct overlay math
  function onImgLoad(e) {
    setImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight });
  }

  return (
    <div className="live-camera-card">
      <div className="live-camera-header">
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div className={`live-dot ${online ? "live-dot--online" : ""}`} />
          <div className="live-camera-name">{cam.name}</div>
        </div>
        <div style={{ display: "flex", align: "center", gap: 8 }}>
          {detections.length > 0 && (
            <span className="badge badge-green">{detections.length} face{detections.length !== 1 ? "s" : ""}</span>
          )}
          {online
            ? <Wifi size={14} color="var(--accent)" />
            : <WifiOff size={14} color="var(--text-muted)" />
          }
        </div>
      </div>

      <div style={{ position: "relative" }}>
        {frame ? (
          <img
            ref={imgRef}
            src={`data:image/jpeg;base64,${frame}`}
            alt={cam.name}
            className="live-img"
            onLoad={onImgLoad}
            style={{ display: "block", width: "100%" }}
          />
        ) : (
          <div style={{
            aspectRatio: "16/9",
            background: "#050c13",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            color: "var(--text-muted)"
          }}>
            <div className="loading-spinner" style={{ width: 24, height: 24, borderWidth: 2 }} />
            <div style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}>Connecting to stream...</div>
          </div>
        )}

        {/* Detection overlays — use actual frame resolution for correct scaling */}
        {detections.map((det, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              left:   `${(det.bbox[0] / imgSize.w) * 100}%`,
              top:    `${(det.bbox[1] / imgSize.h) * 100}%`,
              width:  `${((det.bbox[2] - det.bbox[0]) / imgSize.w) * 100}%`,
              height: `${((det.bbox[3] - det.bbox[1]) / imgSize.h) * 100}%`,
              border: `2px solid ${det.recognized ? "var(--accent)" : "rgba(220,60,60,0.85)"}`,
              pointerEvents: "none",
              boxSizing: "border-box",
            }}
          >
            <div style={{
              position: "absolute",
              bottom: "100%",
              left: 0,
              background: det.recognized ? "var(--accent)" : "rgba(220,60,60,0.92)",
              color: det.recognized ? "#060d14" : "#fff",
              fontSize: 10,
              padding: "1px 6px",
              fontFamily: "var(--font-mono)",
              whiteSpace: "nowrap",
              fontWeight: 600,
            }}>
              {det.name} {det.face_id && `[${det.face_id}]`}
            </div>
          </div>
        ))}
      </div>

      {cam.location && (
        <div style={{
          padding: "6px 12px",
          fontSize: 11,
          fontFamily: "var(--font-mono)",
          color: "var(--text-muted)",
          borderTop: "1px solid var(--border)"
        }}>
          📍 {cam.location}
        </div>
      )}
    </div>
  );
}
