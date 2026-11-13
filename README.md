# 👁️ PaarvAi — Privacy Intelligence System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLO-Detection%20%26%20Tracking-orange?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-PaarvAi%20Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)

**PaarvAi** (derived from the Tamil *Paarvai* — vision / sight) is an enterprise-grade AI computer vision and store intelligence system. It combines real-time object tracking, role-based facial recognition, live CCTV stream analytics, and an interactive dashboard to monitor retail occupancy, identify staff roles, and maintain operational privacy and security.

---

## 🚀 Key Features

### 🎥 Computer Vision & Real-Time Tracking
- **YOLOv8 / YOLO26 Person Detection**: Detects visitors and staff across varied angles and lighting conditions.
- **ByteTrack Multi-Object Tracking**: Stable trajectory estimation and ID persistence across occlusions.
- **Role-Based Face Identification**: Uses OpenCV YuNet (detection) and SFace (deep recognition) to classify recognized individuals:
  - ✨ **Visionary**: Leadership & Store Management
  - 💻 **Hacker**: Technical Operations & Systems
  - ⚡ **Hustler**: Sales, Engagement & Customer Support
  - 👷 **Worker**: General Store Operations
  - 🛍️ **Customer**: Anonymous visitors and shoppers
- **Privacy-Preserving Architecture**: Differentiates between known authorized staff and customer sessions while keeping customer data anonymized with ephemeral session IDs.

### 📊 PaarvAi Interactive Dashboard
- **📹 Live Web Cam Tab**: Direct, real-time camera feed processing with live HUD overlay, person counts, and staff identification alerts.
- **🎥 WhatsApp Video Tab**: Dedicated CCTV video player with:
  - **Live AI Detection Stream**: Real-time YOLO tracking, bounding boxes, role badges, and looping.
  - **Native Video Player**: Embedded HTML5 player with timeline scrub, speed control, and fullscreen.
- **📊 Executive Analytics Tab**: Real-time crowding alerts, behavioral anomaly detection, and customer funnel breakdown.
- **👥 Staff & Role Directory**: Registered employee directory with profiles and assigned roles.
- **📋 Real-Time Event Database**: Filterable database of entry/exit/zone events with one-click CSV export.

### 🔌 Backend & Infrastructure
- **FastAPI REST API**: High-performance API endpoints for events, occupancy, crowding, anomaly, and funnel metrics.
- **SQLite Database Persistence**: Robust storage of store events with graceful offline fallback if FastAPI is offline.
- **Swagger / OpenAPI Documentation**: Auto-generated interactive API docs.
- **Docker & Docker Compose**: Single-command containerized deployment.

---

## 🏗️ System Architecture

```text
 CCTV / Web Cam Stream
         ↓
 YOLOv8 / YOLO26 Person Tracking (ByteTrack)
         ↓
 Face Crop Extraction → YuNet + SFace Recognizer
         ↓
 Role & Identity Classification (Staff vs. Customer)
         ↓
 Event Generation (Entry, Exit, Dwell, Zone)
         ↓
 SQLite Database Persistence
         ↓
 FastAPI REST APIs  ⟷  PaarvAi Streamlit Dashboard
```

---

## 📂 Project Structure

```text
Privacy-Intelligence-System/
├── app/
│   ├── main.py              # FastAPI application & REST endpoints
│   ├── database.py          # SQLAlchemy engine & SQLite configuration
│   ├── models.py            # Pydantic & SQLAlchemy ORM schemas
│   ├── analytics.py         # Customer count & occupancy analytics
│   ├── occupancy.py         # Live occupancy tracking
│   ├── dwell_time.py        # Dwell time calculation
│   ├── crowding.py          # Crowding alert triggers
│   ├── anomaly.py           # Behavioral anomaly detection
│   ├── kpi.py               # Operational KPI computations
│   └── pos.py               # Point-of-Sale transaction metrics
│
├── dashboard/
│   └── app.py               # PaarvAi multi-tab interactive Streamlit dashboard
│
├── pipeline/
│   ├── track_and_identify.py# Main pipeline (YOLO + Face Recognition + DB logging)
│   ├── track_people.py      # Ultralytics YOLO ByteTrack script
│   ├── detect_people.py     # Person detection & video writer
│   ├── face_recognizer.py   # YuNet detection & SFace cosine similarity matcher
│   ├── person_classifier.py # Role classifier (Visionary, Hacker, Hustler, Customer)
│   ├── event_generator.py   # Event logger to SQLite database
│   ├── session_tracker.py   # Visitor session lifecycle tracking
│   └── zones.py             # Store zone definition & tracking
│
├── known_people/            # Registered employee reference portraits
│   ├── Lakshana_EMP-001.jpg
│   ├── Mehaa_EMP-003.jpg
│   ├── Rithika_EMP-005.jpg
│   ├── Shirley_EMP-002.jpg
│   └── Tejas_EMP-004.jpg
│
├── models/                  # Face detection (YuNet) & recognition (SFace) ONNX models
├── sample_data/             # CCTV video samples & POS transactions
├── requirements.txt         # Project dependencies
├── Dockerfile               # Production Docker container definition
├── docker-compose.yml       # Multi-service container orchestration
└── README.md
```

---

## 💻 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Webcam (optional, for live camera tracking)

### 2. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/lakshana-gs/Privacy-Intelligence-System.git
cd Privacy-Intelligence-System
pip install -r requirements.txt
```

### 3. Launching the System

#### Option A: Start Dashboard & Backend
In Terminal 1 (Start FastAPI Backend):
```bash
uvicorn app.main:app --reload --port 8000
```

In Terminal 2 (Start PaarvAi Streamlit Dashboard):
```bash
streamlit run dashboard/app.py
```

- **PaarvAi Dashboard**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Swagger API**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 4. Running the Detection Pipeline via CLI

#### Process CCTV Video File:
```bash
python pipeline/track_and_identify.py --source "sample_data/WhatsApp Video 2026-11-12 at 12.30.21.mp4" --save --output "runs/WhatsApp_Video_result.mp4"
```

#### Run on Live Webcam (Camera Index 0):
```bash
python pipeline/track_and_identify.py --source 0
```

#### Run Ultralytics YOLO Tracking:
```bash
python pipeline/track_people.py "sample_data/WhatsApp Video 2026-11-12 at 12.30.21.mp4"
```

---

## 🐳 Running with Docker

Deploy the complete stack using Docker Compose:

```bash
docker compose up --build
```

Access services:
- **FastAPI**: `http://localhost:8000`
- **Swagger Documentation**: `http://localhost:8000/docs`

---

## 🔌 API Endpoints Reference

### Event Management
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/events` | List all recorded store events |
| `POST` | `/events` | Ingest a new visitor entry/exit event |
| `POST` | `/events/ingest` | Ingest structured telemetry event |

### Analytics & Intelligence
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/metrics` | Total events, entries, and exits |
| `GET` | `/crowding` | Real-time visitor count & crowding alert status |
| `GET` | `/anomaly` | Behavioral anomaly detection analysis |
| `GET` | `/funnel` | In-store customer conversion funnel |
| `GET` | `/kpis` | Operational retail KPIs summary |
| `GET` | `/queue` | Queue completion & abandonment analytics |
| `GET` | `/zones` | Zone entry & dwell time analytics |
| `GET` | `/pos` | POS transactions, revenue, and average basket value |
| `GET` | `/health` | Service and database health check |

---

## 👥 Roles & Category Mapping

| Role | Badge | Description | Examples |
|---|---|---|---|
| **Visionary** | ✨ | Leadership, management, and strategic overview | Tejas (`EMP-004`) |
| **Hacker** | 💻 | Technical operations, system maintenance | Lakshana (`EMP-001`), Mehaa (`EMP-003`) |
| **Hustler** | ⚡ | Sales, customer engagement, POS checkout | Shirley (`EMP-002`), Rithika (`EMP-005`) |
| **Worker** | 👷 | General store associates and staff | Registered Team |
| **Customer** | 🛍️ | Anonymous store visitors and shoppers | Dynamic Track IDs |

---

## 🛡️ Privacy & Security Best Practices
- **Customer Anonymization**: No biometric templates are stored for unregistered visitors; tracking uses ephemeral bounding box track IDs.
- **Local On-Premises Processing**: Computer vision models run entirely locally without transmitting video streams to third-party cloud services.
- **Secure Access**: Credentials and API tokens are excluded from version control via `.gitignore`.

---

## 👩‍💻 Author & Repository

- **Author**: [Lakshana G S](https://github.com/lakshana-gs)
- **Repository**: [https://github.com/lakshana-gs/Privacy-Intelligence-System](https://github.com/lakshana-gs/Privacy-Intelligence-System)
- **License**: MIT License
