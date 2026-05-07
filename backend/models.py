"""FaceGuard Pro - Pydantic models for request/response validation."""

from typing import Optional
from pydantic import BaseModel


class CameraCreate(BaseModel):
    name: str
    rtsp_url: str
    location: Optional[str] = None
    is_active: bool = True


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    location: Optional[str] = None
    is_active: Optional[bool] = None


class Camera(BaseModel):
    id: str
    name: str
    rtsp_url: str
    location: Optional[str] = None
    is_active: bool
    is_streaming: bool = False
    created_at: float

    class Config:
        from_attributes = True


class EmployeeCreate(BaseModel):
    name: str
    department: str
    employee_code: str
    email: Optional[str] = None
    phone: Optional[str] = None


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    employee_code: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class Employee(BaseModel):
    id: str
    face_id: str
    name: str
    department: Optional[str] = None
    employee_code: str
    email: Optional[str] = None
    phone: Optional[str] = None
    image_paths: list[str] = []
    created_at: float

    class Config:
        from_attributes = True


class DetectionEvent(BaseModel):
    id: str
    camera_id: str
    employee_id: Optional[str] = None
    face_id: Optional[str] = None
    employee_name: Optional[str] = None
    confidence: Optional[float] = None
    track_id: Optional[str] = None
    timestamp: float
    frame_b64: Optional[str] = None
