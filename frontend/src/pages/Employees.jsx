import React, { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { getEmployees, addEmployee, updateEmployee, deleteEmployee } from "../api";
import toast from "react-hot-toast";
import {
  Building2,
  Image as ImageIcon,
  Mail,
  Pencil,
  Phone,
  Plus,
  Trash2,
  Upload,
  Users,
  X,
} from "lucide-react";

const DEPARTMENTS = [
  "Engineering",
  "HR",
  "Finance",
  "Operations",
  "Security",
  "Management",
  "IT",
  "Sales",
  "Marketing",
];

const EMPTY_FORM = {
  name: "",
  department: "",
  employee_code: "",
  email: "",
  phone: "",
};

const UPLOADS_BASE = "http://localhost:8000/uploads/";

function buildImageUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${UPLOADS_BASE}${path}`;
}

function initials(name) {
  if (!name) return "?";
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0].toUpperCase())
    .join("");
}

export default function Employees() {
  const [employees, setEmployees] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingEmployee, setEditingEmployee] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [existingPhotos, setExistingPhotos] = useState([]);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    return () => previews.forEach((url) => URL.revokeObjectURL(url));
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

  const onDrop = useCallback((acceptedFiles) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
    setPreviews((prev) => [
      ...prev,
      ...acceptedFiles.map((file) => URL.createObjectURL(file)),
    ]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    multiple: true,
  });

  function resetDraft() {
    setModalOpen(false);
    setEditingEmployee(null);
    setForm(EMPTY_FORM);
    setFiles([]);
    setPreviews([]);
    setExistingPhotos([]);
  }

  function openCreateModal() {
    setEditingEmployee(null);
    setForm(EMPTY_FORM);
    setFiles([]);
    setPreviews([]);
    setExistingPhotos([]);
    setModalOpen(true);
  }

  function openEditModal(emp) {
    setEditingEmployee(emp);
    setForm({
      name: emp.name || "",
      department: emp.department || "",
      employee_code: emp.employee_code || "",
      email: emp.email || "",
      phone: emp.phone || "",
    });
    setFiles([]);
    setPreviews([]);
    setExistingPhotos(emp.image_paths || []);
    setModalOpen(true);
  }

  function removeFile(idx) {
    URL.revokeObjectURL(previews[idx]);
    setFiles((current) => current.filter((_, i) => i !== idx));
    setPreviews((current) => current.filter((_, i) => i !== idx));
  }

  async function save() {
    if (!form.name.trim() || !form.employee_code.trim() || !form.department.trim()) {
      toast.error("Name, Employee Code and Department are required");
      return;
    }

    if (!editingEmployee && files.length === 0) {
      toast.error("Please upload at least one face photo");
      return;
    }

    setSaving(true);
    try {
      const fd = new FormData();
      fd.append("name", form.name.trim());
      fd.append("department", form.department.trim());
      fd.append("employee_code", form.employee_code.trim());
      if (form.email.trim()) fd.append("email", form.email.trim());
      if (form.phone.trim()) fd.append("phone", form.phone.trim());
      files.forEach((file) => fd.append("images", file));

      const emp = editingEmployee
        ? await updateEmployee(editingEmployee.id, fd)
        : await addEmployee(fd);

      setEmployees((prev) =>
        editingEmployee
          ? prev.map((item) => (item.id === emp.id ? emp : item))
          : [emp, ...prev]
      );

      toast.success(
        editingEmployee
          ? `Employee "${emp.name}" updated`
          : `Employee "${emp.name}" registered`
      );
      resetDraft();
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function remove(id, name) {
    if (!window.confirm(`Remove employee "${name}" and their face data?`)) return;

    try {
      await deleteEmployee(id);
      setEmployees((prev) => prev.filter((employee) => employee.id !== id));
      toast.success("Employee removed");
    } catch {
      toast.error("Failed to remove employee");
    }
  }

  const filtered = employees.filter((employee) => {
    const query = search.toLowerCase();
    return (
      employee.name?.toLowerCase().includes(query) ||
      employee.department?.toLowerCase().includes(query) ||
      employee.face_id?.toLowerCase().includes(query) ||
      employee.employee_code?.toLowerCase().includes(query) ||
      employee.email?.toLowerCase().includes(query)
    );
  });

  const isEditMode = Boolean(editingEmployee);

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-title">EMPLOYEES</div>
        <div className="page-sub">Face registration & identity management</div>
      </div>

      <div className="toolbar">
        <div className="toolbar-left">
          <input
            className="input"
            style={{ width: 300 }}
            placeholder="Search by name, dept, code, email..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="toolbar-right">
          <button className="btn btn-primary" onClick={openCreateModal}>
            <Plus size={15} /> Register Employee
          </button>
        </div>
      </div>

      {loading ? (
        <div className="empty-state">
          <div className="loading-spinner" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <Users size={48} className="empty-state-icon" />
          <div className="empty-state-text">
            {search ? "No matches found" : "No employees registered"}
          </div>
          {!search && (
            <button className="btn btn-primary" onClick={openCreateModal}>
              <Plus size={14} /> Register First Employee
            </button>
          )}
        </div>
      ) : (
        <div className="table-wrap employee-table-wrap">
          <table className="employee-table">
            <thead>
              <tr>
                <th>Employee</th>
                <th>Department</th>
                <th>Contact</th>
                <th>Face ID</th>
                <th>Photos</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((employee) => (
                <tr key={employee.id}>
                  <td>
                    <div className="employee-person">
                      {employee.image_paths?.[0] ? (
                        <img
                          src={buildImageUrl(employee.image_paths[0])}
                          alt={employee.name}
                          className="employee-avatar"
                          onError={(event) => {
                            event.currentTarget.style.display = "none";
                          }}
                        />
                      ) : (
                        <div className="employee-avatar employee-avatar--fallback">
                          {initials(employee.name)}
                        </div>
                      )}
                      <div>
                        <div className="employee-name">{employee.name}</div>
                        <div className="employee-subtitle">{employee.employee_code}</div>
                      </div>
                    </div>
                  </td>
                  <td>
                    <div className="employee-value">
                      <Building2 size={14} />
                      {employee.department || "-"}
                    </div>
                  </td>
                  <td>
                    <div className="employee-contact">
                      {employee.email && (
                        <div className="employee-value">
                          <Mail size={14} />
                          {employee.email}
                        </div>
                      )}
                      {employee.phone && (
                        <div className="employee-value">
                          <Phone size={14} />
                          {employee.phone}
                        </div>
                      )}
                      {!employee.email && !employee.phone && <span>-</span>}
                    </div>
                  </td>
                  <td>
                    <span className="badge badge-green">{employee.face_id}</span>
                  </td>
                  <td>
                    <div className="employee-value">
                      <ImageIcon size={14} />
                      {(employee.image_paths || []).length} photo
                      {(employee.image_paths || []).length === 1 ? "" : "s"}
                    </div>
                  </td>
                  <td>
                    <div className="row-actions">
                      <button
                        className="btn btn-sm btn-secondary"
                        onClick={() => openEditModal(employee)}
                      >
                        <Pencil size={12} /> Edit
                      </button>
                      <button
                        className="btn btn-sm btn-danger"
                        onClick={() => remove(employee.id, employee.name)}
                      >
                        <Trash2 size={12} /> Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {modalOpen && (
        <div className="modal-backdrop" onClick={resetDraft}>
          <div className="modal modal--wide" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">
                <Users size={18} style={{ display: "inline", marginRight: 8 }} />
                {isEditMode ? "Edit Employee" : "Register Employee"}
              </div>
              <button className="btn btn-icon btn-secondary" onClick={resetDraft}>
                <X size={16} />
              </button>
            </div>

            <div className="modal-body">
              <div className="form-grid">
                <div className="form-group form-group--full">
                  <label className="form-label">Full Name *</label>
                  <input
                    className="input"
                    placeholder="Rahul Sharma"
                    value={form.name}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, name: event.target.value }))
                    }
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Employee Code *</label>
                  <input
                    className="input"
                    placeholder="EMP001"
                    value={form.employee_code}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, employee_code: event.target.value }))
                    }
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Department *</label>
                  <select
                    className="select"
                    value={form.department}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, department: event.target.value }))
                    }
                  >
                    <option value="">Select...</option>
                    {DEPARTMENTS.map((department) => (
                      <option key={department} value={department}>
                        {department}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Email</label>
                  <input
                    className="input"
                    type="email"
                    placeholder="rahul@company.com"
                    value={form.email}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, email: event.target.value }))
                    }
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Phone</label>
                  <input
                    className="input"
                    placeholder="+91 98765 43210"
                    value={form.phone}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, phone: event.target.value }))
                    }
                  />
                </div>
              </div>

              {isEditMode && existingPhotos.length > 0 && (
                <div className="photo-summary">
                  <div className="form-label">Current Face Photos</div>
                  <div className="img-preview-grid">
                    {existingPhotos.map((path) => (
                      <img
                        key={path}
                        src={buildImageUrl(path)}
                        alt="Current face"
                        className="img-preview"
                      />
                    ))}
                  </div>
                  <div className="helper-text">
                    Uploading new photos will replace the current face set.
                  </div>
                </div>
              )}

              <div className="form-group">
                <label className="form-label">
                  {isEditMode ? "Replace Face Photos" : "Face Photos *"}
                </label>
                <div
                  {...getRootProps()}
                  className={`dropzone ${isDragActive ? "dropzone--active" : ""}`}
                >
                  <input {...getInputProps()} />
                  <div className="dropzone-icon">
                    <Upload size={28} />
                  </div>
                  <div className="dropzone-text">
                    Drop face photos here, or click to browse
                  </div>
                  <div className="dropzone-sub">
                    JPG, PNG, WEBP. Upload multiple angles for better recognition.
                  </div>
                </div>

                {previews.length > 0 && (
                  <div className="img-preview-grid">
                    {previews.map((src, index) => (
                      <div key={src} className="preview-tile">
                        <img src={src} alt="" className="img-preview" />
                        <button
                          type="button"
                          onClick={() => removeFile(index)}
                          className="preview-remove"
                        >
                          <X size={10} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="info-panel">
                Face ID is generated from the employee code as{" "}
                <strong>EMP-{(form.employee_code || "XXX").toUpperCase()}</strong>.
              </div>
            </div>

            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={resetDraft}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={save} disabled={saving}>
                {saving
                  ? "Processing..."
                  : isEditMode
                  ? "Update Employee"
                  : "Register Employee"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
