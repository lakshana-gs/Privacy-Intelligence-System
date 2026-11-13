import os
import sys
import time
import sqlite3
import requests
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure pipeline modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline")))

from ultralytics import YOLO
from pipeline.face_recognizer import FaceIdentifier
from pipeline.person_classifier import PersonClassifier, ROLE_STYLES

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PaarvAi | Retail Vision Intelligence",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: #050816;
    color: #F3F4F6;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 96%;
}

/* Hero Header */
.hero {
    background: linear-gradient(135deg, #3B0764 0%, #6D28D9 50%, #7C3AED 100%);
    border-radius: 20px;
    padding: 22px 32px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(109, 40, 217, 0.45);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.hero-title {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-title span {
    color: #C084FC;
}

.hero-sub {
    color: #E9D5FF;
    font-size: 15px;
    font-weight: 400;
    margin-top: 4px;
}

.live-pill {
    background: rgba(239, 68, 68, 0.2);
    border: 1px solid #EF4444;
    color: #FCA5A5;
    padding: 6px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}

/* KPI Cards */
.kpi-card {
    background: #0F142A;
    border: 1px solid #2D1B69;
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    border-color: #A855F7;
    transform: translateY(-2px);
}

.kpi-icon {
    font-size: 26px;
    margin-bottom: 4px;
}

.kpi-title {
    color: #9CA3AF;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-value {
    color: #FFFFFF;
    font-size: 30px;
    font-weight: 800;
    margin-top: 2px;
}

/* Status Alert Boxes */
.alert-card {
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}

.alert-red {
    background: linear-gradient(180deg, rgba(69, 10, 10, 0.6) 0%, rgba(30, 10, 10, 0.8) 100%);
    border: 1px solid #DC2626;
}

.alert-green {
    background: linear-gradient(180deg, rgba(6, 78, 59, 0.6) 0%, rgba(4, 47, 46, 0.8) 100%);
    border: 1px solid #059669;
}

.alert-title {
    font-size: 16px;
    font-weight: 700;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
}

.alert-desc {
    font-size: 14px;
    color: #D1D5DB;
    margin-top: 6px;
}

/* Staff Card */
.staff-card {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 14px;
    padding: 14px;
    text-align: center;
    margin-bottom: 14px;
}

.staff-role-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA RETRIEVAL (FASTAPI WITH RESILIENT SQLITE FALLBACK)
# -----------------------------------------------------------------------------
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "store.db"))
API_URL = "http://127.0.0.1:8000"

def fetch_system_metrics():
    """Fetch metrics from FastAPI if available, otherwise read directly from store.db."""
    try:
        res_metrics = requests.get(f"{API_URL}/metrics", timeout=0.8).json()
        res_crowding = requests.get(f"{API_URL}/crowding", timeout=0.8).json()
        res_anomaly = requests.get(f"{API_URL}/anomaly", timeout=0.8).json()
        return {
            "source": "FastAPI (Online)",
            "total_events": res_metrics.get("total_events", 0),
            "entries": res_metrics.get("entries", 0),
            "exits": res_metrics.get("exits", 0),
            "visitors": res_crowding.get("current_people", max(0, res_metrics.get("entries", 0) - res_metrics.get("exits", 0))),
            "crowding_alert": res_crowding.get("crowding_alert", False),
            "anomaly": res_anomaly.get("anomaly", False),
            "anomaly_reason": res_anomaly.get("reason", "Normal activity"),
            "backend_online": True
        }
    except Exception:
        pass

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE event_type='entry'")
        entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events WHERE event_type='exit'")
        exits = cursor.fetchone()[0]
        
        active_visitors = max(0, entries - exits)
        crowding_alert = active_visitors > 15
        
        conn.close()
        return {
            "source": "SQLite Engine",
            "total_events": total_events,
            "entries": entries,
            "exits": exits,
            "visitors": active_visitors,
            "crowding_alert": crowding_alert,
            "anomaly": False,
            "anomaly_reason": "Store operational within normal baseline parameters",
            "backend_online": False
        }
    except Exception as e:
        return {
            "source": "Offline Baseline",
            "total_events": 0,
            "entries": 0,
            "exits": 0,
            "visitors": 0,
            "crowding_alert": False,
            "anomaly": False,
            "anomaly_reason": f"No database events recorded ({e})",
            "backend_online": False
        }

def log_event_to_db(event_type, person_id, role="customer"):
    """Save an event directly into store.db."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT, person_id TEXT,
                zone TEXT, role TEXT DEFAULT 'customer',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "INSERT INTO events (event_type, person_id, zone, role, timestamp) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (event_type, person_id, "store", role)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# AI MODEL CACHE (LOADS ONCE)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading PaarvAi Neural Detection & Recognition Engine...")
def load_ai_pipeline(model_path="yolo26n.pt"):
    model = YOLO(model_path)
    face_id = FaceIdentifier()
    classifier = PersonClassifier()
    return model, face_id, classifier

# -----------------------------------------------------------------------------
# HEADER & HERO (PAARVAI BRANDING)
# -----------------------------------------------------------------------------
data = fetch_system_metrics()

col_hero_text, col_hero_badge = st.columns([4, 1])
with col_hero_text:
    st.markdown("""
    <div class="hero">
        <div>
            <div class="hero-title">👁️ PaarvAi <span>Intelligence</span></div>
            <div class="hero-sub">Retail Computer Vision, Live CCTV Tracking & Role-Based AI Analytics</div>
        </div>
        <div class="live-pill">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#EF4444;"></span>
            PAARVAI LIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP METRICS ROW
# -----------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📊</div>
        <div class="kpi-title">Total Events</div>
        <div class="kpi-value">{data['total_events']}</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🚪</div>
        <div class="kpi-title">Store Entries</div>
        <div class="kpi-value">{data['entries']}</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🏃</div>
        <div class="kpi-title">Store Exits</div>
        <div class="kpi-value">{data['exits']}</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">👥</div>
        <div class="kpi-title">Active Visitors</div>
        <div class="kpi-value">{data['visitors']}</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    status_label = "FASTAPI" if data['backend_online'] else "SQLITE"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">💜</div>
        <div class="kpi-title">Engine Mode</div>
        <div class="kpi-value" style="font-size:22px; color:#A78BFA; margin-top:6px;">{status_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# MAIN DASHBOARD TABS: LIVE WEB CAM & WHATSAPP VIDEO
# -----------------------------------------------------------------------------
tab_webcam, tab_whatsapp, tab_kpi, tab_staff, tab_events = st.tabs([
    "📹 Live Web Cam",
    "🎥 WhatsApp Video",
    "📊 Executive Analytics",
    "👥 Staff & Roles",
    "📋 Event Database"
])

# =============================================================================
# TAB 1: LIVE WEB CAM
# =============================================================================
with tab_webcam:
    st.markdown("### 📹 Live Web Cam — Real-Time PaarvAi Vision")
    st.caption("Direct camera feed for real-time customer tracking and instant staff recognition.")

    wc_col_ctrl, wc_col_conf, wc_col_model = st.columns([2, 2, 2])
    with wc_col_ctrl:
        cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1, key="wc_cam_idx")
    with wc_col_conf:
        wc_conf = st.slider("Webcam YOLO Confidence", 0.20, 0.90, 0.35, 0.05, key="wc_conf")
    with wc_col_model:
        wc_model_name = st.selectbox("Webcam Model", ["yolo26n.pt", "yolo26s.pt"], index=0, key="wc_model")

    c_wc_btn1, c_wc_btn2 = st.columns([1.5, 1.5])
    with c_wc_btn1:
        wc_start = st.button("▶️ Start Live Webcam", type="primary", use_container_width=True, key="wc_start")
    with c_wc_btn2:
        wc_stop = st.button("⏹️ Stop Webcam", use_container_width=True, key="wc_stop")

    st.write("---")

    col_wc_stream, col_wc_telemetry = st.columns([7, 3])

    with col_wc_telemetry:
        st.markdown("### 📡 Webcam Telemetry")
        wc_telemetry_box = st.empty()
        wc_telemetry_box.info("Webcam is idle. Click 'Start Live Webcam' to begin.")

        st.markdown("### 🎯 Identified Staff")
        wc_staff_box = st.empty()
        wc_staff_box.write("No staff identified yet.")

    with col_wc_stream:
        wc_frame_placeholder = st.empty()

        if wc_start:
            try:
                model, face_id, classifier = load_ai_pipeline(wc_model_name)
            except Exception as ex:
                st.error(f"Failed to load AI pipeline: {ex}")
                model = None

            if model is not None:
                if sys.platform == "win32":
                    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
                    if not cap.isOpened() or not cap.read()[0]:
                        cap.release()
                        cap = cv2.VideoCapture(int(cam_index))
                else:
                    cap = cv2.VideoCapture(int(cam_index))

                if not cap.isOpened():
                    st.error(f"❌ Cannot connect to Camera Index {cam_index}. Please verify your webcam is connected.")
                else:
                    st.toast("📹 PaarvAi Live Webcam Connected!", icon="👁️")
                    frame_idx = 0
                    logged_identities = set()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Webcam feed interrupted.")
                            break

                        frame_idx += 1

                        results = model.track(
                            frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            classes=[0],
                            conf=wc_conf,
                            verbose=False
                        )

                        active_people = 0
                        detected_staff = []

                        if results and results[0].boxes and results[0].boxes.id is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                            confs = results[0].boxes.conf.cpu().numpy()

                            for box, tid, c in zip(boxes, track_ids, confs):
                                active_people += 1
                                x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                                x2, y2 = min(frame.shape[1], int(box[2])), min(frame.shape[0], int(box[3]))

                                person_crop = frame[y1:y2, x1:x2]
                                face_name = face_id_str = None
                                face_conf = 0.0

                                if frame_idx % 2 == 0:
                                    name, pid, fconf = face_id.recognize_face_in_crop(person_crop, track_id=int(tid))
                                    if name is not None and fconf >= 70.0:
                                        face_name = name
                                        face_id_str = pid
                                        face_conf = fconf

                                person = classifier.update(
                                    track_id=int(tid),
                                    face_name=face_name,
                                    face_id=face_id_str,
                                    face_confidence=face_conf
                                )

                                colour = person.box_color
                                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                                label = person.display_label
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                                cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), colour, -1)
                                cv2.putText(frame, label, (x1 + 5, y1 - 6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                                if person.is_worker and person.face_name:
                                    detected_staff.append({
                                        "name": person.face_name,
                                        "id": person.face_id,
                                        "category": person.category,
                                        "conf": person.face_confidence,
                                        "emoji": person.emoji
                                    })
                                    log_key = f"{tid}_{person.face_name}"
                                    if log_key not in logged_identities:
                                        logged_identities.add(log_key)
                                        log_event_to_db("entry", f"{person.face_name} ({person.face_id}) [{person.category}]", role=person.category.lower())
                                elif person.frames_seen >= 6:
                                    log_key = f"{tid}_customer"
                                    if log_key not in logged_identities:
                                        logged_identities.add(log_key)
                                        log_event_to_db("entry", f"Customer #{tid}", role="customer")

                        # HUD Overlay
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (15, 15, 25), -1)
                        hud_text = f"PaarvAi Live Webcam  |  People: {active_people}  |  Staff: {len(detected_staff)}  |  Frame: #{frame_idx}"
                        cv2.putText(frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 255), 2)

                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        wc_frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        wc_telemetry_box.markdown(f"""
                        <div style="background:#0F172A; border:1px solid #38BDF8; border-radius:12px; padding:14px;">
                            <div style="color:#38BDF8; font-weight:700; font-size:13px;">📡 WEBCAM TELEMETRY</div>
                            <div style="font-size:24px; font-weight:800; color:white; margin:6px 0;">{active_people} <span style="font-size:14px; font-weight:500; color:#94A3B8;">People Visible</span></div>
                            <div style="font-size:13px; color:#CBD5E1;">Live Frame: <b>#{frame_idx}</b></div>
                            <div style="font-size:13px; color:#CBD5E1;">YOLO Confidence: <b>{wc_conf:.2f}</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                        if detected_staff:
                            staff_html = ""
                            for s in detected_staff:
                                staff_html += f"""
                                <div style="background:rgba(109,40,217,0.3); border:1px solid #8B5CF6; border-radius:10px; padding:10px; margin-bottom:8px;">
                                    <div style="font-weight:700; font-size:16px; color:#F5F3FF;">{s['emoji']} {s['name']} <span style="font-size:12px; color:#C4B5FD;">({s['id']})</span></div>
                                    <div style="color:#A78BFA; font-weight:600; font-size:13px; margin-top:2px;">Role: {s['category']} • Match: {s['conf']:.1f}%</div>
                                </div>
                                """
                            wc_staff_box.markdown(staff_html, unsafe_allow_html=True)
                        else:
                            wc_staff_box.markdown("""
                            <div style="background:#1E1B4B; border:1px dashed #6366F1; border-radius:10px; padding:12px; color:#A5B4FC; font-size:13px; text-align:center;">
                                Scanning webcam feed for registered staff...
                            </div>
                            """, unsafe_allow_html=True)

                        time.sleep(0.02)

                    cap.release()
        else:
            wc_frame_placeholder.info("Click 'Start Live Webcam' to stream and track people live from your camera.")

# =============================================================================
# TAB 2: WHATSAPP VIDEO (CCTV FOOTAGE)
# =============================================================================
with tab_whatsapp:
    st.markdown("### 🎥 WhatsApp Video — CCTV Footage Intelligence")
    st.caption("AI person tracking, role classification, and CCTV video playback on WhatsApp Video 2026-11-12 at 12.30.21.mp4.")

    whatsapp_video_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "sample_data", "WhatsApp Video 2026-11-12 at 12.30.21.mp4"
    ))
    tracked_video_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "runs", "WhatsApp_Video_result.mp4"
    ))

    wa_mode_col, wa_ver_col, wa_model_col = st.columns([2, 2, 2])
    with wa_mode_col:
        wa_player_mode = st.radio(
            "Playback Mode",
            ["▶️ Live AI Detection Stream", "🎬 Native CCTV Video Player"],
            horizontal=True,
            key="wa_player_mode"
        )
    with wa_ver_col:
        wa_source_file = st.selectbox(
            "Video Version",
            [
                "Raw WhatsApp Video (WhatsApp Video 2026-11-12 at 12.30.21.mp4)",
                "Saved Tracked Video (runs/WhatsApp_Video_result.mp4)"
            ],
            key="wa_source_file"
        )
    with wa_model_col:
        wa_model_name = st.selectbox("YOLO Model", ["yolo26n.pt", "yolo26s.pt"], index=0, key="wa_model")

    st.write("---")

    selected_video_path = tracked_video_path if "Saved Tracked" in wa_source_file else whatsapp_video_path

    col_wa_video, col_wa_telemetry = st.columns([7, 3])

    with col_wa_telemetry:
        st.markdown("### 📡 CCTV Telemetry")
        wa_telemetry_box = st.empty()
        wa_telemetry_box.info("Ready to stream video. Click 'Start Live Stream' below.")

        st.markdown("### 🎯 Identified In Footage")
        wa_staff_box = st.empty()
        wa_staff_box.write("No staff identified yet in current playback.")

    with col_wa_video:
        if wa_player_mode == "🎬 Native CCTV Video Player":
            st.markdown("#### 🎬 Native Video Player (Playback Controls, Scrubbing, & Fullscreen)")
            if os.path.exists(selected_video_path):
                with open(selected_video_path, "rb") as vid_file:
                    st.video(vid_file.read())
            else:
                st.warning(f"File not found: {selected_video_path}")
        else:
            # Live AI Stream Mode
            st.markdown("#### 🎥 Live AI Detection & Recognition Stream")
            c_wa1, c_wa2, c_wa_conf, c_wa_loop = st.columns([1.5, 1.5, 2, 2])
            with c_wa1:
                wa_start = st.button("▶️ Start Live Stream", type="primary", use_container_width=True, key="wa_start")
            with c_wa2:
                wa_stop = st.button("⏹️ Stop Stream", use_container_width=True, key="wa_stop")
            with c_wa_conf:
                wa_conf_val = st.slider("Detection Confidence", 0.20, 0.90, 0.35, 0.05, key="wa_conf")
            with c_wa_loop:
                wa_loop = st.checkbox("🔄 Continuous Loop", value=True, key="wa_loop")

            wa_video_placeholder = st.empty()

            if wa_start:
                if not os.path.exists(selected_video_path):
                    st.error(f"Cannot find video file: {selected_video_path}")
                else:
                    try:
                        model, face_id, classifier = load_ai_pipeline(wa_model_name)
                    except Exception as ex:
                        st.error(f"Failed to load AI pipeline: {ex}")
                        model = None

                    if model is not None:
                        cap = cv2.VideoCapture(selected_video_path)
                        if not cap.isOpened():
                            st.error(f"Failed to open video: {selected_video_path}")
                        else:
                            st.toast("🎥 PaarvAi CCTV AI Stream Running!", icon="👁️")
                            frame_idx = 0
                            logged_identities = set()

                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    if wa_loop:
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                        continue
                                    else:
                                        break

                                frame_idx += 1

                                results = model.track(
                                    frame,
                                    persist=True,
                                    tracker="bytetrack.yaml",
                                    classes=[0],
                                    conf=wa_conf_val,
                                    verbose=False
                                )

                                active_people = 0
                                detected_staff = []

                                if results and results[0].boxes and results[0].boxes.id is not None:
                                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                                    confs = results[0].boxes.conf.cpu().numpy()

                                    for box, tid, c in zip(boxes, track_ids, confs):
                                        active_people += 1
                                        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                                        x2, y2 = min(frame.shape[1], int(box[2])), min(frame.shape[0], int(box[3]))

                                        person_crop = frame[y1:y2, x1:x2]
                                        face_name = face_id_str = None
                                        face_conf = 0.0

                                        if frame_idx % 2 == 0:
                                            name, pid, fconf = face_id.recognize_face_in_crop(person_crop, track_id=int(tid))
                                            if name is not None and fconf >= 70.0:
                                                face_name = name
                                                face_id_str = pid
                                                face_conf = fconf

                                        person = classifier.update(
                                            track_id=int(tid),
                                            face_name=face_name,
                                            face_id=face_id_str,
                                            face_confidence=face_conf
                                        )

                                        colour = person.box_color
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                                        label = person.display_label
                                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                                        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), colour, -1)
                                        cv2.putText(frame, label, (x1 + 5, y1 - 6),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                                        if person.is_worker and person.face_name:
                                            detected_staff.append({
                                                "name": person.face_name,
                                                "id": person.face_id,
                                                "category": person.category,
                                                "conf": person.face_confidence,
                                                "emoji": person.emoji
                                            })
                                            log_key = f"{tid}_{person.face_name}"
                                            if log_key not in logged_identities:
                                                logged_identities.add(log_key)
                                                log_event_to_db("entry", f"{person.face_name} ({person.face_id}) [{person.category}]", role=person.category.lower())
                                        elif person.frames_seen >= 6:
                                            log_key = f"{tid}_customer"
                                            if log_key not in logged_identities:
                                                logged_identities.add(log_key)
                                                log_event_to_db("entry", f"Customer #{tid}", role="customer")

                                # HUD Overlay
                                cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (15, 15, 25), -1)
                                hud_text = f"PaarvAi CCTV Stream  |  Active Persons: {active_people}  |  Staff: {len(detected_staff)}  |  Frame: #{frame_idx}"
                                cv2.putText(frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 255), 2)

                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                wa_video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                                wa_telemetry_box.markdown(f"""
                                <div style="background:#0F172A; border:1px solid #38BDF8; border-radius:12px; padding:14px;">
                                    <div style="color:#38BDF8; font-weight:700; font-size:13px;">📡 CCTV TELEMETRY</div>
                                    <div style="font-size:24px; font-weight:800; color:white; margin:6px 0;">{active_people} <span style="font-size:14px; font-weight:500; color:#94A3B8;">People in Frame</span></div>
                                    <div style="font-size:13px; color:#CBD5E1;">Processing Frame: <b>#{frame_idx}</b></div>
                                    <div style="font-size:13px; color:#CBD5E1;">YOLO Confidence: <b>{wa_conf_val:.2f}</b></div>
                                </div>
                                """, unsafe_allow_html=True)

                                if detected_staff:
                                    staff_html = ""
                                    for s in detected_staff:
                                        staff_html += f"""
                                        <div style="background:rgba(109,40,217,0.3); border:1px solid #8B5CF6; border-radius:10px; padding:10px; margin-bottom:8px;">
                                            <div style="font-weight:700; font-size:16px; color:#F5F3FF;">{s['emoji']} {s['name']} <span style="font-size:12px; color:#C4B5FD;">({s['id']})</span></div>
                                            <div style="color:#A78BFA; font-weight:600; font-size:13px; margin-top:2px;">Role: {s['category']} • Match: {s['conf']:.1f}%</div>
                                        </div>
                                        """
                                    wa_staff_box.markdown(staff_html, unsafe_allow_html=True)
                                else:
                                    wa_staff_box.markdown("""
                                    <div style="background:#1E1B4B; border:1px dashed #6366F1; border-radius:10px; padding:12px; color:#A5B4FC; font-size:13px; text-align:center;">
                                        Scanning CCTV feed for registered staff...
                                    </div>
                                    """, unsafe_allow_html=True)

                                time.sleep(0.03)

                            cap.release()
            else:
                if os.path.exists(selected_video_path):
                    cap_prev = cv2.VideoCapture(selected_video_path)
                    ret_prev, prev_frame = cap_prev.read()
                    if ret_prev:
                        cv2.rectangle(prev_frame, (0, 0), (prev_frame.shape[1], 46), (15, 15, 25), -1)
                        cv2.putText(prev_frame, "PaarvAi WhatsApp CCTV - Click 'Start Live Stream' to track",
                                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        wa_video_placeholder.image(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    cap_prev.release()

# =============================================================================
# TAB 3: EXECUTIVE ANALYTICS
# =============================================================================
with tab_kpi:
    st.markdown("### 🏬 PaarvAi Store Analytics & Operational KPIs")
    
    col_crowd, col_anom = st.columns(2)
    with col_crowd:
        st.markdown("#### 👥 Crowding Monitoring")
        if data["crowding_alert"]:
            st.markdown(f"""
            <div class="alert-card alert-red">
                <div class="alert-title">🚨 CROWDING ALERT DETECTED</div>
                <div class="alert-desc">Current in-store count ({data['visitors']} visitors) exceeds recommended threshold capacity (15 visitors). Dispatch available staff for customer crowd management.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-card alert-green">
                <div class="alert-title">✅ NORMAL OCCUPANCY</div>
                <div class="alert-desc">Store capacity is well-balanced ({data['visitors']} active visitors). Comfortable browsing experience across aisles.</div>
            </div>
            """, unsafe_allow_html=True)

    with col_anom:
        st.markdown("#### 🛡️ Anomaly Detection")
        if data["anomaly"]:
            st.markdown(f"""
            <div class="alert-card alert-red">
                <div class="alert-title">⚠️ BEHAVIORAL ANOMALY FLAGGED</div>
                <div class="alert-desc">{data['anomaly_reason']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-card alert-green">
                <div class="alert-title">✅ NORMAL STORE PATTERNS</div>
                <div class="alert-desc">{data['anomaly_reason']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("---")

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("#### 🚪 Customer Funnel Breakdown")
        funnel_df = pd.DataFrame({
            "Stage": ["Entered Store", "Active Browsing", "Exited Store"],
            "Count": [data["entries"], data["visitors"], data["exits"]]
        })
        st.bar_chart(funnel_df.set_index("Stage"), color="#8B5CF6")

    with f2:
        st.markdown("#### 📈 Event Activity Timeline")
        try:
            conn = sqlite3.connect(DB_PATH)
            timeline_df = pd.read_sql_query(
                "SELECT timestamp, event_type FROM events ORDER BY id DESC LIMIT 50", conn
            )
            conn.close()
            if not timeline_df.empty:
                timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"])
                timeline_counts = timeline_df.groupby([timeline_df["timestamp"].dt.minute, "event_type"]).size().unstack(fill_value=0)
                st.line_chart(timeline_counts)
            else:
                st.info("No timeline events yet recorded.")
        except Exception:
            st.info("Activity timeline will populate as events are captured.")

# =============================================================================
# TAB 4: STAFF & ROLE DIRECTORY
# =============================================================================
with tab_staff:
    st.markdown("### 👥 PaarvAi Team & Staff Registry")
    st.markdown("Registered store leadership and staff automatically recognized and classified by the Face Recognition engine:")

    staff_members = [
        {"name": "Lakshana", "id": "EMP-001", "role": "Hacker", "desc": "Technical Operations & Store Systems", "img": "known_people/Lakshana_EMP-001.jpg"},
        {"name": "Tejas",    "id": "EMP-004", "role": "Visionary", "desc": "Store Manager & Strategy Lead", "img": "known_people/Tejas_EMP-004.jpg"},
        {"name": "Shirley",  "id": "EMP-002", "role": "Hustler", "desc": "Sales, Customer Engagement & POS Lead", "img": "known_people/Shirley_EMP-002.jpg"},
        {"name": "Mehaa",    "id": "EMP-003", "role": "Hacker", "desc": "Inventory & Technical Support", "img": "known_people/Mehaa_EMP-003.jpg"},
        {"name": "Rithika",  "id": "EMP-005", "role": "Hustler", "desc": "Beauty Advisor & Sales Specialist", "img": "known_people/Rithika_EMP-005.jpg"},
    ]

    cols = st.columns(len(staff_members))
    for col, staff in zip(cols, staff_members):
        with col:
            role_meta = ROLE_STYLES.get(staff["role"], ROLE_STYLES["Worker"])
            bgr = role_meta["color"]
            hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

            img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", staff["img"]))
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.markdown(f"<div style='height:140px; background:#1F2937; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:36px;'>👤</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="staff-card">
                <div style="font-weight:700; font-size:16px; color:white;">{staff['name']}</div>
                <div style="font-size:12px; color:#9CA3AF;">{staff['id']}</div>
                <div class="staff-role-badge" style="background:{hex_color}22; border:1px solid {hex_color}; color:{hex_color};">
                    {role_meta['emoji']} {staff['role']}
                </div>
                <div style="font-size:11px; color:#D1D5DB; margin-top:8px;">{staff['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 5: REAL-TIME EVENT LOGS
# =============================================================================
with tab_events:
    st.markdown("### 📋 PaarvAi Real-Time Store Events Database")
    
    col_filter_type, col_filter_role, col_refresh = st.columns([2, 2, 1])
    
    with col_filter_type:
        event_filter = st.selectbox("Filter by Event Type", ["All Events", "entry", "exit", "zone_entered", "zone_exited"])
    with col_filter_role:
        role_filter = st.selectbox("Filter by Role", ["All Roles", "customer", "hacker", "visionary", "hustler", "worker"])
    with col_refresh:
        st.write("")
        st.write("")
        refresh_btn = st.button("🔄 Refresh Data", use_container_width=True)

    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT id, event_type, person_id, zone, role, timestamp FROM events"
        conditions = []
        if event_filter != "All Events":
            conditions.append(f"event_type = '{event_filter}'")
        if role_filter != "All Roles":
            conditions.append(f"role = '{role_filter}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT 200"

        df_events = pd.read_sql_query(query, conn)
        conn.close()

        if not df_events.empty:
            st.dataframe(df_events, use_container_width=True, height=450)
            csv = df_events.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Event Logs as CSV",
                data=csv,
                file_name="paarvai_store_events.csv",
                mime="text/csv"
            )
        else:
            st.info("No events match the selected filter.")
    except Exception as e:
        st.error(f"Failed to query events table: {e}")
