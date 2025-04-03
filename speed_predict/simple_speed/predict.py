import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch
from datetime import datetime

# Check and set device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Loading Model
model = YOLO("/Users/varun/Documents/yolo_weights/yolov8n.pt")
model.to(device)  # Move model to MPS device

# Loading DeepSort
tracker = DeepSort(max_age=50)  # Increased max_age to handle occlusions better

# Loading video
video_path = "/Users/varun/Documents/gfg_cv/vids_1_yolo/CarsDrivingUnderBridge.mp4"
cap = cv2.VideoCapture(video_path)

# Getting video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Dictionaries for tracking positions and speed history
prev_positions = {}
speed_history = {}
max_history = 5  # Number of frames to average speed over

# Pixel to meter conversion factor (placeholder for calibration)
pixel_to_meter = 0.05  # TODO: Calibrate this value using real-world measurements (e.g., lane width)

# Frame skipping for performance
frame_count = 0
skip_frames = 2  # Process every 2nd frame

# Speed limit
speed_limit = 80  # km/h

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        cv2.imshow("Vehicle Tracking & Speed Estimation", frame)
        out.write(frame)
        continue

    # Add timestamp and speed limit to the frame
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Speed Limit: {speed_limit} km/h", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Run YOLOv8 inference
    results = model.predict(frame, 
                            conf=0.5,          # Higher confidence threshold for tighter boxes
                            iou=0.45,          # IOU threshold for NMS
                            classes=[2, 5, 7], # Only detect cars(2), bus(5), and trucks(7)
                            device=device)     # Use MPS device

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Filter only vehicles (COCO class IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck)
            if cls in [2, 3, 5, 7]:
                detections.append(([x1, y1, x2, y2], conf, cls))

    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    current_time = time.time()

    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box
        x1, y1, x2, y2 = map(int, ltrb)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Default colors for box and text
        box_color = (0, 255, 0)  # Default green
        text_color = (0, 255, 255)  # Default yellow

        # Speed Estimation with Smoothing
        speed_kph = 0  # Default speed
        if track_id in prev_positions:
            prev_x, prev_y, prev_time = prev_positions[track_id]
            distance_pixels = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
            distance_meters = distance_pixels * pixel_to_meter
            time_elapsed = current_time - prev_time

            if time_elapsed > 0:
                speed_mps = distance_meters / time_elapsed
                speed_kph = speed_mps * 3.6  # Convert to km/h

                # Store speed in history for smoothing
                if track_id not in speed_history:
                    speed_history[track_id] = []
                speed_history[track_id].append(speed_kph)
                if len(speed_history[track_id]) > max_history:
                    speed_history[track_id].pop(0)

                # Calculate average speed
                avg_speed_kph = sum(speed_history[track_id]) / len(speed_history[track_id])

                # Log over-speeding vehicles
                if avg_speed_kph > speed_limit:
                    with open("overspeed_log.txt", "a") as f:
                        f.write(f"ID: {track_id}, Speed: {avg_speed_kph:.1f} km/h, Time: {timestamp}\n")
                    box_color = (0, 0, 255)  # Red for over-speed
                    text_color = (0, 0, 255)  # Red text for over-speed

                # Prepare text
                text = f"Speed: {avg_speed_kph:.1f} km/h"
                
                # Add semi-transparent background for speed text
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_w, text_h = text_size
                cv2.rectangle(frame, (x1, y1 - 25 - text_h), (x1 + text_w, y1 - 25), (0, 0, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Update position
        prev_positions[track_id] = (center_x, center_y, current_time)

        # Draw tracking results
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Add ID label with background
        id_text = f"ID: {track_id}"
        id_text_size, _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        id_text_w, id_text_h = id_text_size
        cv2.rectangle(frame, (x1, y1 - 10 - id_text_h), (x1 + id_text_w, y1 - 10), (0, 0, 0), -1)
        cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Show frame
    cv2.imshow("Vehicle Tracking & Speed Estimation", frame)
    out.write(frame)  # Write frame to output video

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()