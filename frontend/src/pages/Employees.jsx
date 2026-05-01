import React, { useEffect, useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { getEmployees, addEmployee, deleteEmployee } from "../api";
import toast from "react-hot-toast";
import { Users, Plus, Trash2, Upload, X, UserCheck } from "lucide-react";

export default function Employees() {
  const [employees, setEmployees] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState(false);
  const [form, setForm] = useState({ name: "", department: "", employee_code: "", email: "", phone: "" });
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => { load(); }, []);

  useEffect(() => {
    return () => previews.forEach(URL.revokeObjectURL);
  }, [previews]);

  async function load() {
    try {
      const data = await getEmployees();
      setEmployees(data);
    } catch {
      toast.error("Failed to load employees");
    } finally {
      setLoading(false);
    }
  }

  const onDrop = useCallback((accepted) => {
    setFiles(prev => [...prev, ...accepted]);
    setPreviews(prev => [...prev, ...accepted.map(f => URL.createObjectURL(f))]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    multiple: true,
  });

  function removeFile(idx) {
    URL.revokeObjectURL(previews[idx]);
    setFiles(f => f.filter((_, i) => i !== idx));
    setPreviews(p => p.filter((_, i) => i !== idx));
  }

  function openModal() {
    setForm({ name: "", department: "", employee_code: "", email: "", phone: "" });
    setFiles([]);
    setPreviews([]);
    setModal(true);
  }

  async function save() {
    if (!form.name.trim() || !form.employee_code.trim() || !form.department.trim()) {
      toast.error("Name, Employee Code and Department are required");
      return;
    }
    if (files.length === 0) {
      toast.error("Please upload at least one face photo");
      return;
    }
    setSaving(true);
    try {
      const fd = new FormData();
      fd.append("name", form.name);
      fd.append("department", form.department);
      fd.append("employee_code", form.employee_code);
      if (form.email) fd.append("email", form.email);
      if (form.phone) fd.append("phone", form.phone);
      files.forEach(f => fd.append("images", f));

      const emp = await addEmployee(fd);
      setEmployees(prev => [emp, ...prev]);
      toast.success(`Employee "${emp.name}" registered · ${emp.face_id}`);
      setModal(false);
    } catch (e) {
      toast.error(e.response?.data?.detail || "Registration failed");
    } finally {
      setSaving(false);
    }
  }

  async function remove(id, name) {
    if (!window.confirm(`Remove employee "${name}" and their face data?`)) return;
    await deleteEmployee(id);
    setEmployees(prev => prev.filter(e => e.id !== id));
    toast.success("Employee removed");
  }

  const filtered = employees.filter(e =>
    e.name.toLowerCase().includes(search.toLowerCase()) ||
    e.department?.toLowerCase().includes(search.toLowerCase()) ||
    e.face_id.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-title">EMPLOYEES</div>
        <div className="page-sub">Face registration & identity management</div>
      </div>

      <div className="toolbar">
        <div className="toolbar-left">
          <input className="input" style={{ width: 260 }} placeholder="Search by name, dept, face ID..."
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="toolbar-right">
          <button className="btn btn-primary" onClick={openModal}>
            <Plus size={15} /> Register Employee
          </button>
        </div>
      </div>

      {loading ? (
        <div className="empty-state"><div className="loading-spinner" /></div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <Users size={48} className="empty-state-icon" />
          <div className="empty-state-text">{search ? "No matches found" : "No employees registered"}</div>
          {!search && <button className="btn btn-primary" onClick={openModal}><Plus size={14} /> Register First Employee</button>}
        </div>
      ) : (
        <div className="employee-grid">
          {filtered.map(emp => (
            <div className="employee-card" key={emp.id}>
              {emp.image_paths && emp.image_paths[0] ? (
                <img
                  src={`http://localhost:8000/uploads/${emp.image_paths[0]}`}
                  alt={emp.name}
                  className="emp-avatar"
                  onError={e => { e.target.style.display = "none"; }}
                />
              ) : (
                <div className="emp-avatar-placeholder">{emp.name[0]}</div>
              )}
              <div className="emp-name">{emp.name}</div>
              <div className="emp-dept">{emp.department}</div>
              <div className="emp-faceid">{emp.face_id}</div>
              {emp.image_paths && emp.image_paths.length > 0 && (
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 10, fontFamily: "var(--font-mono)" }}>
                  {emp.image_paths.length} face angle{emp.image_paths.length !== 1 ? "s" : ""}
                </div>
              )}
              {emp.email && (
                <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 8 }}>{emp.email}</div>
              )}
              <div className="emp-actions">
                <button className="btn btn-sm btn-danger" onClick={() => remove(emp.id, emp.name)}>
                  <Trash2 size={12} /> Remove
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(false)}>
          <div className="modal" style={{ maxWidth: 620 }} onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title"><UserCheck size={18} style={{ display: "inline", marginRight: 8 }} />Register Employee</div>
              <button className="btn btn-icon btn-secondary" onClick={() => setModal(false)}><X size={16} /></button>
            </div>
            <div className="modal-body">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <div className="form-group" style={{ gridColumn: "1/-1" }}>
                  <label className="form-label">Full Name *</label>
                  <input className="input" placeholder="Rahul Sharma" value={form.name}
                    onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
                </div>
                <div className="form-group">
                  <label className="form-label">Employee Code *</label>
                  <input className="input" placeholder="EMP001" value={form.employee_code}
                    onChange={e => setForm(f => ({ ...f, employee_code: e.target.value }))} />
                </div>
                <div className="form-group">
                  <label className="form-label">Department *</label>
                  <select className="select" value={form.department}
                    onChange={e => setForm(f => ({ ...f, department: e.target.value }))}>
                    <option value="">Select...</option>
                    {["Engineering", "HR", "Finance", "Operations", "Security", "Management", "IT", "Sales", "Marketing"].map(d => (
                      <option key={d} value={d}>{d}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Email</label>
                  <input className="input" type="email" placeholder="rahul@company.com" value={form.email}
                    onChange={e => setForm(f => ({ ...f, email: e.target.value }))} />
                </div>
                <div className="form-group">
                  <label className="form-label">Phone</label>
                  <input className="input" placeholder="+91 98765 43210" value={form.phone}
                    onChange={e => setForm(f => ({ ...f, phone: e.target.value }))} />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Face Photos * (multiple angles recommended)</label>
                <div {...getRootProps()} className={`dropzone ${isDragActive ? "dropzone--active" : ""}`}>
                  <input {...getInputProps()} />
                  <div className="dropzone-icon"><Upload size={28} /></div>
                  <div className="dropzone-text">Drop face photos here, or click to browse</div>
                  <div className="dropzone-sub">JPG, PNG, WEBP · Upload from different angles for better accuracy</div>
                </div>
                {previews.length > 0 && (
                  <div className="img-preview-grid">
                    {previews.map((src, i) => (
                      <div key={i} style={{ position: "relative" }}>
                        <img src={src} alt="" className="img-preview" />
                        <button
                          onClick={() => removeFile(i)}
                          style={{
                            position: "absolute", top: -6, right: -6,
                            width: 18, height: 18, borderRadius: "50%",
                            background: "var(--red)", border: "none",
                            color: "#fff", cursor: "pointer", fontSize: 10,
                            display: "flex", alignItems: "center", justifyContent: "center"
                          }}
                        ><X size={10} /></button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div style={{
                background: "var(--accent-dim)", border: "1px solid rgba(0,229,160,0.2)",
                borderRadius: "var(--radius)", padding: "10px 14px",
                fontSize: 12, color: "var(--text-secondary)", fontFamily: "var(--font-mono)"
              }}>
                💡 Face ID will be auto-generated as <strong style={{ color: "var(--accent)" }}>EMP-{(form.employee_code || "XXX").toUpperCase()}</strong>.
                Upload photos from multiple angles (front, left, right) for best recognition accuracy.
              </div>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={save} disabled={saving}>
                {saving ? "Processing..." : "Register Employee"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
