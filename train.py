import cv2
import torch
import numpy as np
import threading
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("ultralytics/yolov5", "yolov5n", device=device)  # You can choose different model sizes (e.g., 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\ram\OneDrive\Documents\164tbp\Advanced-Aerial-Drone-Detection-System\best.pt').to(device)

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)


cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)


cap1.set(cv2.CAP_PROP_FPS, 30)
cap2.set(cv2.CAP_PROP_FPS, 30)

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

KNOWN_DISTANCE_BETWEEN_CAMERAS = 30 # cm
FOCAL_LENGTH = 485 # Pixels
camera_distance_above_ground = 0

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


thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

while True:
    with lock:
        if frame1 is None or frame2 is None:
            continue
        img1 = frame1.copy()
        img2 = frame2.copy()

    img1 = cv2.resize(img1, (FRAME_WIDTH, FRAME_HEIGHT))
    img2 = cv2.resize(img2, (FRAME_WIDTH, FRAME_HEIGHT))

    CENTER_X, CENTER_Y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    cv2.drawMarker(img1, (CENTER_X, CENTER_Y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.drawMarker(img2, (CENTER_X, CENTER_Y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.putText(img1, f"Center: ({CENTER_X}, {CENTER_Y})", 
                (CENTER_X - 100, CENTER_Y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(img2, f"Center: ({CENTER_X}, {CENTER_Y})", 
                (CENTER_X - 100, CENTER_Y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    
    start_time = time.time()

    
    results1 = model(img1)
    results2 = model(img2)

    x1, y1, x2, y2 = -1, -1, -1, -1
    x3, y3, x4, y4 = -1, -1, -1, -1
    obj_center_x, obj_center_y = -1, -1

    
    detections_cam1 = []
    detections_cam2 = []


    for result in results1.xyxy[0]:  
        class_id = int(result[5])  
        class_name = model.names[class_id]  
        print(f"Camera 1 detected: {class_name}")  
        if class_name == 'chair':  
            x1, y1, x2, y2 = map(int, result[:4])
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            detections_cam1.append(((x1, y1, x2, y2), (obj_center_x, obj_center_y)))
            cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img1, f"Object Center: ({obj_center_x}, {obj_center_y})", 
                        (obj_center_x, obj_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(img1, f"LT: ({x1}, {y1})", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img1, f"LB: ({x1}, {y2})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    
    for result in results2.xyxy[0]:
        class_id = int(result[5])  
        class_name = model.names[class_id] 
        print(f"Camera 2 detected: {class_name}")  
        if class_name == 'chair':  
            x3, y3, x4, y4 = map(int, result[:4])
            obj_center_x = (x3 + x4) // 2
            obj_center_y = (y3 + y4) // 2
            detections_cam2.append(((x3, y3, x4, y4), (obj_center_x, obj_center_y)))
            cv2.rectangle(img2, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(img2, f"Object Center: ({obj_center_x}, {obj_center_y})", 
                        (obj_center_x, obj_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    
    if detections_cam1 and detections_cam2:
        for (box1, center1) in detections_cam1:
            min_dist = float("inf")
            best_match = None
            for (box2, center2) in detections_cam2:
                y_diff = abs(center1[1] - center2[1])
                if y_diff < 50:  
                    x_disparity = abs(center1[0] - center2[0])
                    if x_disparity < min_dist:
                        min_dist = x_disparity
                        best_match = (box2, center2)

            if best_match:
                (x1, y1, x2, y2) = box1
                obj_center_x1, obj_center_y1 = center1
                (x3, y3, x4, y4) = best_match[0]
                obj_center_x2, obj_center_y2 = best_match[1]
                #depth=Z * (240 - obj_center_y1) / 410 + camera_distance_above_ground
                # Calculate disparity and depth
                disparity = abs(obj_center_x1 - obj_center_x2)
                Z = (KNOWN_DISTANCE_BETWEEN_CAMERAS * FOCAL_LENGTH) / disparity if disparity > 0 else 0
                depth = abs(obj_center_y1 - CENTER_Y)
                cv2.putText(img1, f"Distance: {Z:.2f} cm", (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img2, f"Distance: {Z:.2f} cm", (x3 - 10, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img1, f"Depth: {Z * (240 - obj_center_y1) / 647 + camera_distance_above_ground:.2f} cm", 
                            (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #cv2.putText(img1,f"left: {Z * (320 - obj_center_x)/500}cm", 
                 #    q       (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #cv2.putText(img2, f"Depth: {depth:.2f} cm", (x3, y3 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(img1, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img2, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    
    cv2.imshow("Camera 1", img1)
    cv2.imshow("Camera 2", img2)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap1.release()
cap2.release()
