import cv2
import torch
import numpy as np
import threading
from ultralytics import YOLO

# Load YOLOv8 model on GPU for faster processing
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt").to(device)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\ram\OneDrive\Documents\164tbp\Advanced-Aerial-Drone-Detection-System\best.pt', source='github')

# IP Camera URLs
# RTSP Camera URLs
URL_1 = "rtsp://192.168.0.156:8080/h264_pcm.sdp"
URL_2 = "rtsp://192.168.0.105:8082/h264_pcm.sdp"  # Replace with correct IP of 2nd phone

# Open video streams
cap1 = cv2.VideoCapture(URL_1)
cap2 = cv2.VideoCapture(URL_2)

# Reduce buffer delay
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Reduce FPS to 30 for smoother real-time processing
cap1.set(cv2.CAP_PROP_FPS, 30)
cap2.set(cv2.CAP_PROP_FPS, 30)

# Set resolution
FRAME_WIDTH, FRAME_HEIGHT = 640,480
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Camera parameters for depth estimation
KNOWN_DISTANCE_BETWEEN_CAMERAS = 93 # cm
FOCAL_LENGTH = 485 # Pixels

# Frame storage for multi-threading
frame1 = None
frame2 = None
lock = threading.Lock()

def capture_frames():
    global frame1, frame2
    while True:
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        
        if ret1 and ret2:
            with lock:
                frame1 = img1
                frame2 = img2

# Start frame capture thread
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

while True:
    with lock:
        if frame1 is None or frame2 is None:
            continue

        img1 = frame1.copy()
        img2 = frame2.copy()

    # Resize for faster processing
    img1 = cv2.resize(img1, (FRAME_WIDTH, FRAME_HEIGHT))
    img2 = cv2.resize(img2, (FRAME_WIDTH, FRAME_HEIGHT))

    # Display frame center
    CENTER_X, CENTER_Y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    cv2.drawMarker(img1, (CENTER_X, CENTER_Y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.drawMarker(img2, (CENTER_X, CENTER_Y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.putText(img1, f"Center: ({CENTER_X}, {CENTER_Y})", 
                (CENTER_X - 100, CENTER_Y ), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.putText(img2, f"Center: ({CENTER_X}, {CENTER_Y})", 
                (CENTER_X - 100, CENTER_Y ), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Perform YOLOv8 object detection
    results1 = model(img1, conf=0.4)
    results2 = model(img2, conf=0.4)
    camera_distance_above_ground=62.8
    # Process detections
    x1, y1, x2, y2 = -1, -1, -1, -1


    for result in results2:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            if class_name == 'Drone':
                x3, y3, x4, y4 = map(int, box.xyxy[0])
                obj_center_x = (x3 + x4) // 2
                obj_center_y = (y3 + y4) // 2
                
                cv2.rectangle(img2, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cv2.putText(img2, f"Object Center: ({obj_center_x}, {obj_center_y})", 
                            (obj_center_x, obj_center_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
    for result in results1:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            if class_name == 'Drone':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2
                
                cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img1, f"Object Center: ({obj_center_x}, {obj_center_y})", 
                            (obj_center_x, obj_center_y ), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(img1, f"LT: ({x1}, {y1})", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img1, f"LB: ({x1}, {y2})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
    # Calculate Distance and Depth
    if x1 != -1 and x3 != -1:
        disparity = abs(x1 - x3)
        if disparity > 0:
            Z = (KNOWN_DISTANCE_BETWEEN_CAMERAS * FOCAL_LENGTH) / disparity
        else:
            Z = 0  # Avoid division by zero

        # Correct Depth Calculation
        depth = abs(obj_center_y - CENTER_Y)

        # Display values
        cv2.putText(img1, f"Distance: {Z:.2f} cm", (x1-10, y1 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img2, f"Distance: {Z:.2f} cm", (x3-10, y3 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #cv2.putText(img1, f"height: {Z*abs(y1-y2)/647:.2f} cm", (x1-20, y1 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 155, 0), 2)
#depth from camera axis .if from ground add height where camera is kept
        cv2.putText(img1, f"Depth: {Z*(240-obj_center_y)/647+camera_distance_above_ground:.2f} cm", 
                    (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(img2, f"Depth: {depth:.2f} cm", 
                    (x3, y3 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(img1, (obj_center_x, obj_center_y), 5, (0, 0, 255), -1)
    
    combined_frame = np.hstack((img1, img2))

    # Show the combined feed
    cv2.imshow("bottle", combined_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
