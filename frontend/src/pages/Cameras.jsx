import React, { useEffect, useState } from "react";
import { getCameras, addCamera, updateCamera, deleteCamera, toggleCamera } from "../api";
import toast from "react-hot-toast";
import {
  Camera, Plus, Trash2, Edit2, Power, PowerOff, MapPin, X, WifiOff
} from "lucide-react";

const EMPTY = { name: "", rtsp_url: "", location: "", is_active: true };

export default function Cameras() {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState(null); // null | "add" | "edit"
  const [form, setForm] = useState(EMPTY);
  const [editId, setEditId] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => { load(); }, []);

  async function load() {
    try {
      const data = await getCameras();
      setCameras(data);
    } catch (e) {
      toast.error("Failed to load cameras");
    } finally {
      setLoading(false);
    }
  }

  function openAdd() {
    setForm(EMPTY);
    setEditId(null);
    setModal("add");
  }

  function openEdit(cam) {
    setForm({ name: cam.name, rtsp_url: cam.rtsp_url, location: cam.location || "", is_active: cam.is_active });
    setEditId(cam.id);
    setModal("edit");
  }

  async function save() {
    if (!form.name.trim() || !form.rtsp_url.trim()) {
      toast.error("Name and RTSP URL required");
      return;
    }
    setSaving(true);
    try {
      if (modal === "add") {
        const cam = await addCamera(form);
        setCameras(prev => [cam, ...prev]);
        toast.success(`Camera "${cam.name}" added`);
      } else {
        const cam = await updateCamera(editId, form);
        setCameras(prev => prev.map(c => c.id === editId ? cam : c));
        toast.success("Camera updated");
      }
      setModal(null);
    } catch (e) {
      toast.error(e.response?.data?.detail || "Error saving camera");
    } finally {
      setSaving(false);
    }
  }

  async function remove(id, name) {
    if (!window.confirm(`Delete camera "${name}"?`)) return;
    await deleteCamera(id);
    setCameras(prev => prev.filter(c => c.id !== id));
    toast.success("Camera deleted");
  }

  async function toggle(id) {
    const result = await toggleCamera(id);
    setCameras(prev => prev.map(c => c.id === id ? { ...c, is_active: result.status === "started", is_streaming: result.status === "started" } : c));
    toast.success(result.status === "started" ? "Stream started" : "Stream stopped");
  }

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-title">CAMERAS</div>
        <div className="page-sub">Manage RTSP camera streams</div>
      </div>

      <div className="toolbar">
        <div className="toolbar-left">
          <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>
            {cameras.length} camera{cameras.length !== 1 ? "s" : ""} · {cameras.filter(c => c.is_streaming).length} active
          </span>
        </div>
        <div className="toolbar-right">
          <button className="btn btn-primary" onClick={openAdd}>
            <Plus size={15} /> Add Camera
          </button>
        </div>
      </div>

      {loading ? (
        <div className="empty-state"><div className="loading-spinner" /></div>
      ) : cameras.length === 0 ? (
        <div className="empty-state">
          <Camera size={48} className="empty-state-icon" />
          <div className="empty-state-text">No cameras configured</div>
          <button className="btn btn-primary" onClick={openAdd}><Plus size={14} /> Add First Camera</button>
        </div>
      ) : (
        <div className="camera-grid">
          {cameras.map((cam) => (
            <div className="camera-card" key={cam.id}>
              <div className="camera-feed">
                {cam.is_streaming ? (
                  <div style={{ color: "var(--accent)", fontSize: 12, fontFamily: "var(--font-mono)", textAlign: "center" }}>
                    <div className="pulse-dot" style={{ margin: "0 auto 8px" }} />
                    LIVE STREAM ACTIVE
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
                      Open Live Feed tab to view
                    </div>
                  </div>
                ) : (
                  <div className="camera-feed-offline">
                    <WifiOff size={28} />
                    <div style={{ fontSize: 12, fontFamily: "var(--font-mono)" }}>OFFLINE</div>
                  </div>
                )}
              </div>
              <div className="camera-info">
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
                  <div className="camera-name">{cam.name}</div>
                  <span className={`badge ${cam.is_streaming ? "badge-green" : "badge-red"}`}>
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: "currentColor" }} />
                    {cam.is_streaming ? "Online" : "Offline"}
                  </span>
                </div>
                <div className="camera-url">{cam.rtsp_url}</div>
                {cam.location && (
                  <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 4, fontSize: 12, color: "var(--text-muted)" }}>
                    <MapPin size={11} /> {cam.location}
                  </div>
                )}
                <div className="camera-actions">
                  <button
                    className={`btn btn-sm ${cam.is_active ? "btn-danger" : "btn-primary"}`}
                    onClick={() => toggle(cam.id)}
                  >
                    {cam.is_active ? <PowerOff size={13} /> : <Power size={13} />}
                    {cam.is_active ? "Stop" : "Start"}
                  </button>
                  <button className="btn btn-sm btn-secondary" onClick={() => openEdit(cam)}>
                    <Edit2 size={13} /> Edit
                  </button>
                  <button className="btn btn-sm btn-danger" onClick={() => remove(cam.id, cam.name)}>
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">{modal === "add" ? "Add Camera" : "Edit Camera"}</div>
              <button className="btn btn-icon btn-secondary" onClick={() => setModal(null)}><X size={16} /></button>
            </div>
            <div className="modal-body">
              <div className="form-group">
                <label className="form-label">Camera Name</label>
                <input className="input" placeholder="e.g. Main Entrance" value={form.name}
                  onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">RTSP URL</label>
                <input className="input" placeholder="rtsp://192.168.1.10:554/stream" value={form.rtsp_url}
                  onChange={e => setForm(f => ({ ...f, rtsp_url: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">Location (optional)</label>
                <input className="input" placeholder="e.g. Building A, Floor 2" value={form.location}
                  onChange={e => setForm(f => ({ ...f, location: e.target.value }))} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <input type="checkbox" id="is_active" checked={form.is_active}
                  onChange={e => setForm(f => ({ ...f, is_active: e.target.checked }))}
                  style={{ accentColor: "var(--accent)", width: 14, height: 14 }} />
                <label htmlFor="is_active" style={{ fontSize: 13, color: "var(--text-secondary)", cursor: "pointer" }}>
                  Start stream immediately
                </label>
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setModal(null)}>Cancel</button>
              <button className="btn btn-primary" onClick={save} disabled={saving}>
                {saving ? "Saving..." : modal === "add" ? "Add Camera" : "Save Changes"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
