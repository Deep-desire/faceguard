import axios from "axios";

const API = axios.create({ baseURL: "http://localhost:8000/api" });

// ── Cameras ────────────────────────────────────────────────────────────────
export const getCameras    = ()         => API.get("/cameras").then(r => r.data);
export const addCamera     = (data)     => API.post("/cameras", data).then(r => r.data);
export const updateCamera  = (id, data) => API.put(`/cameras/${id}`, data).then(r => r.data);
export const deleteCamera  = (id)       => API.delete(`/cameras/${id}`);
export const toggleCamera  = (id)       => API.post(`/cameras/${id}/toggle`).then(r => r.data);

// ── Employees ──────────────────────────────────────────────────────────────
export const getEmployees   = ()         => API.get("/employees").then(r => r.data);
export const addEmployee    = (formData) => API.post("/employees", formData, {
  headers: { "Content-Type": "multipart/form-data" },
}).then(r => r.data);
export const updateEmployee = (id, formData) => API.put(`/employees/${id}`, formData, {
  headers: { "Content-Type": "multipart/form-data" },
}).then(r => r.data);
export const deleteEmployee = (id)       => API.delete(`/employees/${id}`);

// ── Dashboard ──────────────────────────────────────────────────────────────
export const getStats  = () => API.get("/dashboard/stats").then(r => r.data);
export const getEvents = (limit = 100, camera_id) => {
  const params = { limit };
  if (camera_id) params.camera_id = camera_id;
  return API.get("/events", { params }).then(r => r.data);
};

// ── WebSocket helpers ──────────────────────────────────────────────────────
export const WS_BASE = "ws://localhost:8000";

export function createCameraWS(camId, onMessage, onClose) {
  const ws = new WebSocket(`${WS_BASE}/ws/camera/${camId}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose   = onClose || (() => {});
  return ws;
}

export function createEventsWS(onMessage) {
  const ws = new WebSocket(`${WS_BASE}/ws/events`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return ws;
}
