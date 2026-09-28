import streamlit as st
import cv2
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(page_title="AeroGuard-3K C2 Dashboard", layout="wide")

st.title("🛡️ AeroGuard-3K: Command & Control Center")
st.caption("Edge-Based Autonomous Anti-Drone Defense System")

# 2. Database Functions
def load_db_data():
    conn = sqlite3.connect("known_drones.db")
    df = pd.read_sql_query("SELECT * FROM drones", conn)
    conn.close()
    return df

def check_iff_status(drone_id):
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM drones WHERE drone_id = ?", (drone_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "UNKNOWN_FOE"

# Helper function to add a new drone directly via sidebar
def register_new_drone(drone_id, owner, status):
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO drones (drone_id, owner, status)
        VALUES (?, ?, ?)
    """, (drone_id, owner, status))
    conn.commit()
    conn.close()

def log_threat_event(drone_id):
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            drone_id TEXT,
            status TEXT
        )
    """)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO threat_logs (timestamp, drone_id, status) VALUES (?, ?, ?)",
                   (now, drone_id, "FOE"))
    conn.commit()
    conn.close()

def load_threat_logs():
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='threat_logs'")
    if cursor.fetchone():
        df = pd.read_sql_query("SELECT timestamp, drone_id, status FROM threat_logs ORDER BY id DESC LIMIT 5", conn)
        conn.close()
        return df
    conn.close()
    return pd.DataFrame(columns=["timestamp", "drone_id", "status"])

# 3. Sidebar: Live Database Management
st.sidebar.header("⚙️ Target Registration")
new_id = st.sidebar.text_input("Drone ID (e.g., DRONE_003)")
new_owner = st.sidebar.text_input("Owner / Unit")
new_status = st.sidebar.selectbox("IFF Status", ["FRIEND", "FOE"])

if st.sidebar.button("Register Target"):
    if new_id and new_owner:
        register_new_drone(new_id, new_owner, new_status)
        st.sidebar.success(f"Registered {new_id} successfully!")
    else:
        st.sidebar.error("Please provide both Drone ID and Owner.")

# 4. Main Dashboard Layout
alert_placeholder = st.empty()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Video Feed & Threat Detection")
    run_feed = st.checkbox("Start Live Feed", value=False)
    frame_placeholder = st.empty()

with col2:
    st.subheader("📡 Registered Target Database")
    db_table_placeholder = st.empty()
    db_table_placeholder.dataframe(load_db_data(), use_container_width=True)
    
    st.subheader("🚨 Recent Threat Logs")
    logs_placeholder = st.empty()
    logs_placeholder.dataframe(load_threat_logs(), use_container_width=True)

# 5. Processing Loop
if run_feed:
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    frame_counter = 0

    while cap.isOpened() and run_feed:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break

        results = model(frame, stream=True)
        active_threat_detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                test_drone_id = "DRONE_001" if cls_id == 0 else "DRONE_002"
                status = check_iff_status(test_drone_id)
                
                if status != "FRIEND":
                    active_threat_detected = True
                    if frame_counter % 30 == 0:
                        log_threat_event(test_drone_id)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if status == "FRIEND" else (0, 0, 255)
                label = f"{test_drone_id}: {status}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if active_threat_detected:
            alert_placeholder.error("🚨 THREAT ALERT: Unregistered Object / FOE Detected in Sector!")
        else:
            alert_placeholder.success("✅ SYSTEM SECURE: No Threats Detected")

        if frame_counter % 30 == 0:
            db_table_placeholder.dataframe(load_db_data(), use_container_width=True)
            logs_placeholder.dataframe(load_threat_logs(), use_container_width=True)

        frame_counter += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()