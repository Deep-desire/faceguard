import React from "react";
import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import Dashboard from "./pages/Dashboard";
import Cameras from "./pages/Cameras";
import Employees from "./pages/Employees";
import LiveFeed from "./pages/LiveFeed";
import Events from "./pages/Events";
import {
  LayoutDashboard, Camera, Users, MonitorPlay, Activity, Shield
} from "lucide-react";
import "./index.css";

const NAV = [
  { to: "/dashboard", label: "Dashboard",  icon: LayoutDashboard },
  { to: "/cameras",   label: "Cameras",    icon: Camera          },
  { to: "/employees", label: "Employees",  icon: Users           },
  { to: "/live",      label: "Live Feed",  icon: MonitorPlay     },
  { to: "/events",    label: "Events",     icon: Activity        },
];

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="sidebar-brand">
            <Shield size={26} className="brand-icon" />
            <div>
              <div className="brand-name">FaceGuard</div>
              <div className="brand-sub">Pro Intelligence</div>
            </div>
          </div>

          <nav className="sidebar-nav">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `nav-item ${isActive ? "nav-item--active" : ""}`
                }
              >
                <Icon size={18} />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="sidebar-footer">
            <div className="sys-status">
              <span className="pulse-dot" />
              System Online
            </div>
          </div>
        </aside>

        {/* ── Main Content ── */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/cameras"   element={<Cameras />}   />
            <Route path="/employees" element={<Employees />} />
            <Route path="/live"      element={<LiveFeed />}  />
            <Route path="/events"    element={<Events />}    />
          </Routes>
        </main>
      </div>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#0f1923",
            color: "#e2e8f0",
            border: "1px solid #1e3a4a",
            fontFamily: "var(--font-mono)",
            fontSize: "13px",
          },
          success: { iconTheme: { primary: "#00e5a0", secondary: "#0f1923" } },
          error:   { iconTheme: { primary: "#ff4757", secondary: "#0f1923" } },
        }}
      />
    </BrowserRouter>
  );
}
