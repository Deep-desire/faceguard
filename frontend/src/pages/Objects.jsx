import React, { useEffect, useMemo, useState } from "react";
import { getObjectDetections, getCameras } from "../api";
import { PackageSearch, Filter, Camera, User, Clock3, Image as ImageIcon, Layers3 } from "lucide-react";
import { format, fromUnixTime } from "date-fns";

export default function Objects() {
  const [objects, setObjects] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState("");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    load();
  }, [cameraId]);

  async function load() {
    setLoading(true);
    const [items, cams] = await Promise.all([
      getObjectDetections(200, cameraId || undefined),
      cameras.length ? Promise.resolve(cameras) : getCameras(),
    ]);
    setObjects(items);
    setCameras(cams);
    setLoading(false);
  }

  const filtered = useMemo(() => {
    const term = query.trim().toLowerCase();
    if (!term) return objects;
    return objects.filter((item) => {
      const haystack = [
        item.object_id,
        item.object_label,
        item.employee_name,
        item.employee_id,
        item.camera_id,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(term);
    });
  }, [objects, query]);

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-title">OBJECT INVENTORY</div>
        <div className="page-sub">Stable object IDs, linked employees, and persisted snapshots</div>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <div className="card-header" style={{ marginBottom: 0 }}>
          <div className="card-title">Filters</div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <div className="input-shell" style={{ minWidth: 260 }}>
              <Filter size={16} style={{ color: "var(--text-muted)" }} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search object ID, label, employee..."
                className="input"
              />
            </div>
            <select className="input" value={cameraId} onChange={(e) => setCameraId(e.target.value)} style={{ minWidth: 180 }}>
              <option value="">All cameras</option>
              {cameras.map((cam) => (
                <option key={cam.id} value={cam.id}>{cam.name}</option>
              ))}
            </select>
            <span className="badge badge-green">{filtered.length} tracks</span>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="empty-state" style={{ minHeight: 240 }}>
          <div className="loading-spinner" />
          <div className="empty-state-text" style={{ marginTop: 12 }}>Loading object tracks...</div>
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state" style={{ minHeight: 260 }}>
          <PackageSearch size={48} className="empty-state-icon" />
          <div className="empty-state-text">No object detections yet for the current filters.</div>
        </div>
      ) : (
        <div className="objects-grid">
          {filtered.map((item) => (
            <div key={item.object_id} className="card object-card">
              <div className="object-thumb-wrap">
                {item.snapshot_path ? (
                  <img
                    src={`http://localhost:8000/uploads/${item.snapshot_path}`}
                    alt={item.object_label}
                    className="object-thumb"
                  />
                ) : (
                  <div className="object-thumb object-thumb--empty">
                    <ImageIcon size={28} />
                  </div>
                )}
                <div className="object-chip">{item.object_label}</div>
              </div>

              <div className="object-meta">
                <div className="object-title">{item.object_id}</div>
                <div className="object-row"><Camera size={14} /> {item.camera_id}</div>
                <div className="object-row"><Layers3 size={14} /> Seen {item.occurrence_count || 1} times</div>
                <div className="object-row"><Clock3 size={14} /> {formatTs(item.last_seen)}</div>
                <div className="object-row"><User size={14} /> {item.employee_name || "Unlinked"}{item.employee_id ? ` (${item.employee_id})` : ""}</div>
                {item.confidence != null && (
                  <div className="object-row"><PackageSearch size={14} /> {(item.confidence * 100).toFixed(0)}% confidence</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatTs(ts) {
  if (!ts) return "—";
  try {
    return format(fromUnixTime(ts), "dd MMM yyyy · HH:mm:ss");
  } catch {
    return "—";
  }
}
