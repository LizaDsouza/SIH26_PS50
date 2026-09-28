
import cv2
import sqlite3
from ultralytics import YOLO

# 1. Load YOLO model
model = YOLO("yolov8n.pt")

# 2. Function to check IFF status in SQLite database
def determine_alert_level(category, confidence):
    category = category.lower()
    
    if category == 'drone':
        if confidence >= 0.70:
            return 'CRITICAL'
        else:
            return 'Medium' 
            
    elif category == 'person':
        return 'Medium' 
        
    else: # Birds, animals, or general clutter
        return 'Low' 

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
            status = determine_alert_level(test_drone_id, box.conf[0])
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