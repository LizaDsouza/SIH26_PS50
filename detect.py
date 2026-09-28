# import cv2
# from ultralytics import YOLO

# # 1. Load the pre-trained YOLOv8 nano model
# model = YOLO("yolov8n.pt")

# # 2. Open webcam (0) or replace "0" with a video file path e.g. "drone_test.mp4"
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLOv8 on the current frame
#     results = model(frame, stream=True)

#     # Visualize results on the frame
#     for r in results:
#         annotated_frame = r.plot()
#         cv2.imshow("AeroGuard-3K - Real-Time Detection", annotated_frame)

#     # Press 'q' on your keyboard to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import sqlite3
from ultralytics import YOLO

# 1. Load YOLO model
model = YOLO("yolov8n.pt")

# 2. Function to check IFF status in SQLite database
def check_iff_status(drone_id):
    conn = sqlite3.connect("known_drones.db")
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM drones WHERE drone_id = ?", (drone_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    return "UNKNOWN_FOE"

# 3. Open video stream (Webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # For testing: map class 0 (person) to DRONE_001 (FRIEND) 
            # and other classes to DRONE_002 (FOE)
            cls_id = int(box.cls[0])
            test_drone_id = "DRONE_001" if cls_id == 0 else "DRONE_002"
            
            # Query local DB
            status = check_iff_status(test_drone_id)
            
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Choose color: Green for FRIEND, Red for FOE
            color = (0, 255, 0) if status == "FRIEND" else (0, 0, 255)
            
            # Draw box & label on frame
            label = f"{test_drone_id}: {status}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("AeroGuard-3K - IFF Threat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()