import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch
from datetime import datetime
import argparse

# Parse command-line arguments for input source
def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Speed Detection System")
    parser.add_argument("--source", type=str, default="/Users/varun/Documents/gfg_cv/vids_1_yolo/CarsDrivingUnderBridge.mp4",
                        help="Input source: path to video/image, or '0' for webcam")
    return parser.parse_args()

# Initialize input source (video, image, or webcam)
def initialize_source(source):
    try:
        if source == "0":  # Webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("Could not access webcam. Ensure it is connected and not in use.")
            return cap, False  # False indicates not a single image
        elif source.lower().endswith(('.jpg', '.jpeg', '.png')):  # Image
            frame = cv2.imread(source)
            if frame is None:
                raise ValueError(f"Could not load image from {source}.")
            return frame, True  # True indicates a single image
        else:  # Video
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {source}.")
            return cap, False
    except Exception as e:
        print(f"Error initializing source: {e}")
        exit(1)

# Main processing function
def process_frame(frame, model, tracker, prev_positions, speed_history, max_history, pixel_to_meter, speed_limit):
    current_time = time.time()
    results = model.predict(frame, conf=0.5, iou=0.45, classes=[2, 5, 7], device=device)
    detections = []

    # Extract detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:
                detections.append(([x1, y1, x2, y2], conf, cls))

    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Process each tracked object
    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Initialize default colors
        box_color = (0, 255, 0)  # Default green
        text_color = (0, 255, 255)  # Default yellow
        speed_text = "Speed: N/A"

        # Speed estimation with smoothing
        if track_id in prev_positions:
            prev_x, prev_y, prev_time = prev_positions[track_id]
            distance_pixels = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
            distance_meters = distance_pixels * pixel_to_meter
            time_elapsed = current_time - prev_time

            if time_elapsed > 0:
                speed_mps = distance_meters / time_elapsed
                speed_kph = speed_mps * 3.6

                if track_id not in speed_history:
                    speed_history[track_id] = []
                speed_history[track_id].append(speed_kph)
                if len(speed_history[track_id]) > max_history:
                    speed_history[track_id].pop(0)

                avg_speed_kph = sum(speed_history[track_id]) / len(speed_history[track_id])

                # Log over-speeding vehicles
                if avg_speed_kph > speed_limit:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("overspeed_log.txt", "a") as f:
                        f.write(f"ID: {track_id}, Speed: {avg_speed_kph:.1f} km/h, Time: {timestamp}\n")

                # Update text and colors based on speed
                speed_text = f"Speed: {avg_speed_kph:.1f} km/h"
                if avg_speed_kph > speed_limit:
                    text_color = (0, 0, 255)  # Red for overspeeding
                    box_color = (0, 0, 255)

        prev_positions[track_id] = (center_x, center_y, current_time)

        # Add semi-transparent background for speed text
        text_size, _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x1, y1 - 25 - text_h), (x1 + text_w, y1 - 25), (0, 0, 0), -1)
        cv2.putText(frame, speed_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        id_text = f"ID: {track_id}"
        id_text_size, _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        id_text_w, id_text_h = id_text_size
        cv2.rectangle(frame, (x1, y1 - 10 - id_text_h), (x1 + id_text_w, y1 - 10), (0, 0, 0), -1)
        cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame

# Main function
def main():
    args = parse_args()
    source = args.source

    # Check and set device
    global device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and tracker
    try:
        model = YOLO("/Users/varun/Documents/yolo_weights/yolov8n.pt")
        model.to(device)
        tracker = DeepSort(max_age=50)
    except Exception as e:
        print(f"Error loading model or tracker: {e}")
        exit(1)

    # Initialize input source
    source_obj, is_image = initialize_source(source)

    # Initialize tracking variables
    prev_positions = {}
    speed_history = {}
    max_history = 5
    pixel_to_meter = 0.05  # TODO: Calibrate this value
    speed_limit = 80
    frame_count = 0
    skip_frames = 2

    # Main loop
    try:
        if is_image:
            frame = source_obj
            frame = process_frame(frame, model, tracker, prev_positions, speed_history, max_history, pixel_to_meter, speed_limit)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Speed Limit: {speed_limit} km/h", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Vehicle Tracking & Speed Estimation", frame)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        else:
            while source_obj.isOpened():
                ret, frame = source_obj.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    cv2.imshow("Vehicle Tracking & Speed Estimation", frame)
                    if cv2.waitKey(30) & 0xFF == ord("q"):  # Add 30ms delay
                        break
                    continue

                frame = process_frame(frame, model, tracker, prev_positions, speed_history, max_history, pixel_to_meter, speed_limit)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Speed Limit: {speed_limit} km/h", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Vehicle Tracking & Speed Estimation", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):  # Add 30ms delay
                    break

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        if not is_image:
            source_obj.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()