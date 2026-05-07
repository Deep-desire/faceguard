# FaceGuard Pro Guidebook

## Product Guidebook

### Overview
FaceGuard Pro is a state-of-the-art CCTV face recognition system designed for real-time surveillance. It combines cutting-edge technologies like YOLOv8 for face detection, ArcFace for face recognition, and DeepSORT for object tracking to deliver accurate and efficient face recognition capabilities. The system is tailored for security and monitoring applications in industries such as corporate offices, retail, and public safety.

---

### Features

#### 1. Camera Management
- **Purpose**: Manage RTSP camera streams.
- **Workflow**:
  1. Add, edit, or delete cameras.
  2. Toggle camera streams on/off.
- **Entry Points**:
  - Frontend: `Cameras.jsx`
  - Backend: `/api/cameras` endpoints.
- **APIs**:
  - `GET /api/cameras`: List all cameras.
  - `POST /api/cameras`: Add a new camera.
  - `PUT /api/cameras/{id}`: Update camera details.
  - `DELETE /api/cameras/{id}`: Remove a camera.
  - `POST /api/cameras/{id}/toggle`: Start/stop a camera stream.

#### 2. Employee Registration
- **Purpose**: Register employees with face photos for recognition.
- **Workflow**:
  1. Fill in employee details (name, code, department).
  2. Upload multiple face photos from different angles.
  3. Auto-generate Face ID.
- **Entry Points**:
  - Frontend: `Employees.jsx`
  - Backend: `/api/employees` endpoints.
- **APIs**:
  - `GET /api/employees`: List all employees.
  - `POST /api/employees`: Register a new employee.
  - `PUT /api/employees/{id}`: Update employee details.
  - `DELETE /api/employees/{id}`: Remove an employee.

#### 3. Live Feed
- **Purpose**: Monitor real-time camera streams and face recognition.
- **Workflow**:
  1. Display active camera streams.
  2. Highlight recognized faces with bounding boxes and names.
  3. Show unknown faces with blue bounding boxes.
- **Entry Points**:
  - Frontend: `LiveFeed.jsx`
  - Backend: `/ws/camera/{cam_id}` WebSocket.
- **APIs**:
  - `WS /ws/camera/{cam_id}`: Stream live frames and detections.
  - `WS /ws/events`: Stream real-time detection events.

#### 4. Event History
- **Purpose**: View historical detection logs.
- **Workflow**:
  1. Filter events by camera or recognition status.
  2. View face snapshots, confidence scores, and timestamps.
- **Entry Points**:
  - Frontend: `Events.jsx`
  - Backend: `/api/events` endpoints.
- **APIs**:
  - `GET /api/events`: Retrieve detection history.
  - `GET /api/dashboard/stats`: Fetch dashboard statistics.

---

### User Guide

#### 1. Adding Cameras
1. Navigate to the **Cameras** tab.
2. Click **Add Camera**.
3. Enter the camera name, RTSP URL, and location.
4. Toggle the stream on/off.

#### 2. Registering Employees
1. Navigate to the **Employees** tab.
2. Click **Register Employee**.
3. Fill in the employee details.
4. Upload multiple face photos.
5. Save the registration.

#### 3. Monitoring Live Feed
1. Navigate to the **Live Feed** tab.
2. View active camera streams.
3. Recognized faces appear with green bounding boxes.
4. Unknown faces appear with blue bounding boxes.

#### 4. Viewing Event History
1. Navigate to the **Events** tab.
2. Filter events by camera or recognition status.
3. View snapshots and details of past detections.

---

## Developer Guidebook

### System Architecture

#### High-Level Overview
- **Frontend**: React application for user interaction.
- **Backend**: FastAPI for API and WebSocket endpoints.
- **Database**: SQLite for data persistence.
- **Deployment**: Dockerized for portability.

#### Data Flow
1. RTSP Frame → Process every 3rd frame.
2. Quality Gate → Filters based on blur, size, and angle.
3. ArcFace Embedding → Generates 512-d embeddings.
4. Cosine Similarity → Matches embeddings with a threshold of 0.55.
5. Temporal Vote Buffer → Smooths recognition over 5/10 frames.
6. DeepSORT → Tracks bounding boxes.

#### Component Interaction
- **Frontend**: Handles user interactions and displays real-time data.
- **Backend**: Processes video streams, manages database, and serves APIs.
- **Database**: Stores employee data, camera configurations, and event logs.

---

### Backend Details

#### Core Modules
- `main.py`: API server and WebSocket endpoints.
- `face_engine.py`: YOLOv8 + ArcFace + Vote Buffer pipeline.
- `camera_manager.py`: RTSP stream management.
- `database.py`: SQLite persistence layer.
- `models.py`: Pydantic request/response models.

#### Key Algorithms
- **Face Detection**: YOLOv8n-face model.
- **Face Recognition**: ArcFace embeddings with cosine similarity.
- **Object Tracking**: DeepSORT for bounding box tracking.

#### API Endpoints
- **Cameras**:
  - `GET /api/cameras`
  - `POST /api/cameras`
  - `PUT /api/cameras/{id}`
  - `DELETE /api/cameras/{id}`
- **Employees**:
  - `GET /api/employees`
  - `POST /api/employees`
  - `PUT /api/employees/{id}`
  - `DELETE /api/employees/{id}`
- **Events**:
  - `GET /api/events`
  - `GET /api/dashboard/stats`

---

### Frontend Details

#### Core Components
- `App.jsx`: Router and sidebar layout.
- `api.js`: Axios and WebSocket helpers.
- `Dashboard.jsx`: Displays stats and recent events.
- `Cameras.jsx`: Manages RTSP cameras.
- `Employees.jsx`: Registers employees.
- `LiveFeed.jsx`: Displays real-time streams.
- `Events.jsx`: Shows detection history.

#### State Management
- **Library**: React Context API.
- **Usage**: Manages global state for cameras, employees, and events.

---

### Deployment Guide

#### Docker
1. Build the backend image:
   ```bash
   docker build -t faceguard-backend ./backend
   ```
2. Run the backend container:
   ```bash
   docker run -p 8000:8000 -v ./data:/app/data faceguard-backend
   ```
3. Build the frontend:
   ```bash
   cd frontend && npm run build
   ```
4. Serve the frontend with Nginx.

#### Environment Variables
- `DATABASE_PATH`: Path to SQLite database.
- `UPLOAD_DIR`: Directory for uploaded files.
- `COSINE_THRESHOLD`: Similarity threshold for recognition.
- `FRAME_SKIP`: Number of frames to skip.
- `VOTE_WINDOW`: Temporal smoothing window.
- `VOTE_THRESHOLD`: Minimum votes for recognition.

---

### Testing Guide

#### Unit Tests
- **Backend**: Use `pytest` for testing API endpoints and core logic.
- **Frontend**: Use `Jest` and `React Testing Library` for component testing.

#### Integration Tests
- Test API and database interactions.
- Test WebSocket communication.

#### End-to-End Tests
- Use `Cypress` for testing user workflows.

---

### Future Improvements

#### Security Enhancements
- Add OAuth2 authentication.
- Implement input validation and sanitization.

#### Performance Optimizations
- Introduce GPU acceleration for all models.
- Implement caching for frequently accessed data.

#### Scalability
- Transition to a distributed database.
- Introduce microservices for independent scaling.

---

## Conclusion
FaceGuard Pro is a robust and scalable system for real-time face recognition. This guidebook serves as a comprehensive reference for both users and developers, ensuring smooth operation, maintenance, and future development.