import streamlit as st
import cv2
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(page_title="AeroGuard-3K C2 Dashboard", layout="wide")

st.title("AeroGuard-3K: Command & Control Center")
st.caption("Edge-Based Autonomous Multi-Class Threat Defense System")

# 2. Database Functions
def fetch_registered_entities():
    """Fetches static registered profiles (drones, security personnel, etc.)."""
    conn = sqlite3.connect("known_drones.db")
    # Using fallback table check in case schema is transitioning
    try:
        df = pd.read_sql_query("SELECT * FROM entities", conn)
    except Exception:
        df = pd.read_sql_query("SELECT * FROM drones", conn)
    conn.close()
    return df

def register_new_entity(name, category, status):
    """Registers a known profile into the database."""
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    cursor.execute("""
        INSERT INTO entities (name, category, status)
        VALUES (?, ?, ?)
    """, (name, category, status))
    conn.commit()
    conn.close()

def log_detection_event(category, confidence, bbox, alert_level, location="Sector A"):
    """Logs real-time detection telemetry into the detections table."""
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            bounding_box TEXT,
            location TEXT,
            alert_level TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO detections (category, confidence, bounding_box, location, alert_level)
        VALUES (?, ?, ?, ?, ?)
    """, (category, round(confidence, 2), str(bbox), location, alert_level))
    conn.commit()
    conn.close()

def fetch_detection_logs():
    """Retrieves live detection logs from the database."""
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
    if cursor.fetchone():
        df = pd.read_sql_query("""
            SELECT detection_id, category, confidence, timestamp, alert_level, location 
            FROM detections ORDER BY detection_id DESC LIMIT 50
        """, conn)
        conn.close()
        return df
    conn.close()
    return pd.DataFrame(columns=["detection_id", "category", "confidence", "timestamp", "alert_level", "location"])

def determine_alert_level(category, confidence):
    """Assigns security priority based on object category and confidence score."""
    cat = category.lower()
    if "drone" in cat:
        return "CRITICAL" if confidence >= 0.60 else "Medium"
    elif "person" in cat:
        return "Medium"
    else:  # Bird or biological noise
        return "Low"

# 3. Sidebar: Target Registration & Data Filters
st.sidebar.header("Entity Registration")
new_name = st.sidebar.text_input("Name / Label (e.g., Patrol Unit 1, DJI-01)")
new_cat = st.sidebar.selectbox("Category", ["Drone", "Person", "Bird"])
new_status = st.sidebar.selectbox("Status", ["Authorized", "Unauthorized", "Neutral"])

if st.sidebar.button("Register Target"):
    if new_name:
        register_new_entity(new_name, new_cat, new_status)
        st.sidebar.success(f"Registered {new_name} ({new_cat}) successfully!")
    else:
        st.sidebar.error("Please enter a target name/label.")

st.sidebar.markdown("---")
st.sidebar.header("Filter Live Logs")
filter_category = st.sidebar.multiselect("Category:", ["Drone", "Person", "Bird"], default=["Drone", "Person", "Bird"])
filter_alert = st.sidebar.multiselect("Alert Level:", ["CRITICAL", "Medium", "Low"], default=["CRITICAL", "Medium", "Low"])

# 4. Main Dashboard Layout
alert_placeholder = st.empty()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Real-Time Threat Surveillance")
    run_feed = st.checkbox("Start Live Feed", value=False)
    frame_placeholder = st.empty()

with col2:
    st.subheader("Fleet & Entity Database")
    db_table_placeholder = st.empty()
    db_table_placeholder.dataframe(fetch_registered_entities(), use_container_width=True)

st.markdown("---")
st.subheader("Real-Time Threat Telemetry Logs")

# Summary Metric Cards
m1, m2, m3 = st.columns(3)
metric_total = m1.empty()
metric_crit = m2.empty()
metric_drones = m3.empty()

logs_placeholder = st.empty()

# 5. Live Processing Loop
if run_feed:
    model = YOLO("yolov8n.pt")  # COCO model (Class 0: person, Class 14: bird)
    cap = cv2.VideoCapture(0)
    frame_counter = 0

    while cap.isOpened() and run_feed:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera stream.")
            break

        results = model(frame, stream=True)
        active_critical_alert = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]

                # Map YOLO class names to our domain entities
                if class_name == "person":
                    category = "Person"
                elif class_name == "bird":
                    category = "Bird"
                else:
                    # Generic placeholder for demo: treats unlisted moving items as Drones
                    category = "Drone"

                alert_level = determine_alert_level(category, conf)

                if alert_level == "CRITICAL":
                    active_critical_alert = True

                # Log to database every 30 frames (~1 sec) to prevent database lock
                if frame_counter % 30 == 0:
                    bbox = [int(c) for c in box.xyxy[0]]
                    log_detection_event(category, conf, bbox, alert_level)

                # Draw Bounding Boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if alert_level == "CRITICAL" else (0, 255, 255) if alert_level == "Medium" else (0, 255, 0)
                label = f"{category} ({int(conf*100)}%) [{alert_level}]"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update Alert Banner
        if active_critical_alert:
            alert_placeholder.error("CRITICAL THREAT DETECTED: Unauthorized Aerial / Perimeter Intrusion!")
        else:
            alert_placeholder.success("SYSTEM SECURE: Sector Clear")

        # Update Data Tables & Metrics periodically
        if frame_counter % 15 == 0:
            df_logs = fetch_detection_logs()
            
            # Apply Sidebar Filters
            if not df_logs.empty:
                filtered_df = df_logs[
                    (df_logs["category"].isin(filter_category)) & 
                    (df_logs["alert_level"].isin(filter_alert))
                ]
            else:
                filtered_df = df_logs

            logs_placeholder.dataframe(filtered_df, use_container_width=True)
            db_table_placeholder.dataframe(fetch_registered_entities(), use_container_width=True)

            # Update Metrics
            metric_total.metric("Total Detections", len(filtered_df))
            metric_crit.metric("Critical Alerts", len(filtered_df[filtered_df["alert_level"] == "CRITICAL"]) if not filtered_df.empty else 0)
            metric_drones.metric("Drones Spotted", len(filtered_df[filtered_df["category"] == "Drone"]) if not filtered_df.empty else 0)

        frame_counter += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()