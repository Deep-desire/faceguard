import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getStats, getEvents } from "../api";
import {
  Camera, Users, Activity, Eye, TrendingUp, Clock, PackageSearch
} from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { format, fromUnixTime } from "date-fns";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [events, setEvents] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 10000);
    return () => clearInterval(id);
  }, []);

  async function fetchAll() {
    const [s, e] = await Promise.all([getStats(), getEvents(50)]);
    setStats(s);
    setEvents(e);
  }

  // Build hourly chart data from events
  const chartData = buildHourlyData(events);

  const STAT_CARDS = stats ? [
    { label: "Total Cameras",   value: stats.total_cameras,        icon: Camera,   color: "#3b9eff", to: "/cameras" },
    { label: "Active Streams",  value: stats.active_cameras,       icon: Eye,      color: "#00e5a0", to: "/live"    },
    { label: "Employees",       value: stats.total_employees,      icon: Users,    color: "#a78bfa", to: "/employees" },
    { label: "Events Today",    value: stats.events_today,         icon: Activity, color: "#ffa502", to: "/events" },
    { label: "Unique Detections",value: stats.unique_detections_today,icon:TrendingUp,color:"#ff6b6b", to: "/events" },
    { label: "Objects Today",   value: stats.objects_today,        icon: PackageSearch, color: "#f59e0b", to: "/objects" },
    { label: "Registered Faces",value: stats.registered_faces,     icon: Users,    color: "#00e5a0", to: "/employees" },
  ] : [];

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-title">COMMAND CENTER</div>
        <div className="page-sub">
          {new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })} · IST
        </div>
      </div>

      {/* Stat Cards */}
      <div className="stats-grid">
        {STAT_CARDS.map((s) => (
          <div
            className="stat-card stat-card--clickable"
            key={s.label}
            style={{ "--stat-color": s.color }}
            onClick={() => navigate(s.to)}
          >
            <div className="stat-icon"><s.icon size={22} /></div>
            <div className="stat-value">{s.value ?? "—"}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="section-grid">
        {/* Detection Chart */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Detection Activity (24h)</div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="accentGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#00e5a0" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00e5a0" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <XAxis dataKey="hour" tick={{ fill: "#3a5570", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#3a5570", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: "#0f1e2e", border: "1px solid #1a3248", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "#7a9ab5" }}
                itemStyle={{ color: "#00e5a0" }}
              />
              <Area type="monotone" dataKey="count" stroke="#00e5a0" fill="url(#accentGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Events */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Recent Detections</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, maxHeight: 220, overflowY: "auto" }}>
            {events.slice(0, 8).map((ev) => (
              <div key={ev.id} className="event-item">
                {ev.frame_b64 ? (
                  <img src={`data:image/jpeg;base64,${ev.frame_b64}`} alt="" className="event-face-img" />
                ) : (
                  <div className="event-face-img" style={{ background: "var(--bg-surface)", display:"flex", alignItems:"center", justifyContent:"center" }}>
                    <Users size={16} color="var(--text-muted)" />
                  </div>
                )}
                <div>
                  <div className="event-name">{ev.employee_name || "Unknown"}</div>
                  <div className="event-detail">
                    {ev.face_id || "—"} · {ev.confidence ? `${(ev.confidence * 100).toFixed(0)}%` : ""} · {formatTs(ev.timestamp)}
                  </div>
                </div>
              </div>
            ))}
            {events.length === 0 && (
              <div className="empty-state" style={{ padding: "30px" }}>
                <Clock size={28} className="empty-state-icon" />
                <div className="empty-state-text">No detections yet</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function buildHourlyData(events) {
  const counts = {};
  for (let h = 0; h < 24; h++) counts[h] = 0;
  const now = Date.now() / 1000;
  events.forEach((ev) => {
    const age = now - ev.timestamp;
    if (age < 86400) {
      const h = new Date(ev.timestamp * 1000).getHours();
      counts[h] = (counts[h] || 0) + 1;
    }
  });
  return Object.entries(counts).map(([h, count]) => ({
    hour: `${h.padStart(2, "0")}:00`,
    count,
  }));
}

function formatTs(ts) {
  try {
    return format(fromUnixTime(ts), "HH:mm:ss");
  } catch {
    return "—";
  }
}
