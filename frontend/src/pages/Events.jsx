import React, { useEffect, useState } from "react";
import { getEvents, getCameras } from "../api";
import { Activity, Filter, RefreshCw, User } from "lucide-react";
import { format, fromUnixTime } from "date-fns";

export default function Events() {
  const [events, setEvents] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [filter, setFilter] = useState({ camera_id: "", recognized: "all" });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getCameras().then(setCameras);
    load();
  }, []);

  async function load() {
    setLoading(true);
    try {
      const data = await getEvents(200, filter.camera_id || undefined);
      setEvents(data);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, [filter.camera_id]);

  const filtered = events.filter(ev => {
    if (filter.recognized === "recognized") return ev.face_id != null;
    if (filter.recognized === "unknown") return ev.face_id == null;
    return true;
  });

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-title">DETECTION EVENTS</div>
        <div className="page-sub">Historical face detection log</div>
      </div>

      <div className="toolbar">
        <div className="toolbar-left">
          <Filter size={15} style={{ color: "var(--text-muted)" }} />
          <select className="select" style={{ width: 200 }}
            value={filter.camera_id} onChange={e => setFilter(f => ({ ...f, camera_id: e.target.value }))}>
            <option value="">All Cameras</option>
            {cameras.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
          <select className="select" style={{ width: 160 }}
            value={filter.recognized} onChange={e => setFilter(f => ({ ...f, recognized: e.target.value }))}>
            <option value="all">All Detections</option>
            <option value="recognized">Recognized Only</option>
            <option value="unknown">Unknown Only</option>
          </select>
        </div>
        <div className="toolbar-right">
          <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            {filtered.length} events
          </span>
          <button className="btn btn-secondary btn-sm" onClick={load}>
            <RefreshCw size={13} /> Refresh
          </button>
        </div>
      </div>

      <div className="card">
        {loading ? (
          <div className="empty-state"><div className="loading-spinner" /></div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <Activity size={40} className="empty-state-icon" />
            <div className="empty-state-text">No events found</div>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Face</th>
                  <th>Employee</th>
                  <th>Face ID</th>
                  <th>Camera</th>
                  <th>Confidence</th>
                  <th>Track ID</th>
                  <th>Timestamp</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(ev => {
                  const cam = cameras.find(c => c.id === ev.camera_id);
                  return (
                    <tr key={ev.id}>
                      <td>
                        {ev.frame_b64 ? (
                          <img
                            src={`data:image/jpeg;base64,${ev.frame_b64}`}
                            alt=""
                            style={{ width: 40, height: 40, objectFit: "cover", borderRadius: 6, border: "1px solid var(--border)" }}
                          />
                        ) : (
                          <div style={{
                            width: 40, height: 40, borderRadius: 6,
                            background: "var(--bg-surface)", display: "flex",
                            alignItems: "center", justifyContent: "center"
                          }}>
                            <User size={16} color="var(--text-muted)" />
                          </div>
                        )}
                      </td>
                      <td style={{ fontWeight: 500 }}>{ev.employee_name || <span style={{ color: "var(--text-muted)" }}>Unknown</span>}</td>
                      <td>
                        {ev.face_id
                          ? <span className="badge badge-green">{ev.face_id}</span>
                          : <span className="badge badge-blue">Unregistered</span>
                        }
                      </td>
                      <td>{cam?.name || ev.camera_id?.slice(0, 8) + "..."}</td>
                      <td>
                        {ev.confidence
                          ? <ConfBar value={ev.confidence} />
                          : <span style={{ color: "var(--text-muted)" }}>—</span>
                        }
                      </td>
                      <td>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>
                          {ev.track_id?.split("_").pop() || "—"}
                        </span>
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                        {format(fromUnixTime(ev.timestamp), "dd MMM HH:mm:ss")}
                      </td>
                      <td>
                        <span className={`badge ${ev.face_id ? "badge-green" : "badge-amber"}`}>
                          {ev.face_id ? "✓ Match" : "? Unknown"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function ConfBar({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "var(--accent)" : pct >= 60 ? "var(--amber)" : "var(--red)";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 60, height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2 }} />
      </div>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color }}>{pct}%</span>
    </div>
  );
}
