import numpy as np
import cv2
import torch
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv

# Lane detection functions
def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

def detect_lanes(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    
    # Hough transform parameters
    rho = 1
    theta = np.pi/180
    threshold = 20
    min_line_length = 20
    max_line_gap = 500
    
    lines = cv2.HoughLinesP(region, rho, theta, threshold, 
                           minLineLength=min_line_length, 
                           maxLineGap=max_line_gap)
    
    return lines if lines is not None else []

def calculate_lane_positions(frame, lines):
    if len(lines) == 0:
        return None
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
    
    return left_lines, right_lines

class VehicleTracker:
    def __init__(self, fps):
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))
        self.lane_positions = defaultdict(lambda: deque(maxlen=fps))
        self.fps = fps
        
    def update(self, detections, frame_width):
        speeds = {}
        lane_changes = {}
        
        for idx, (tracker_id, bbox) in enumerate(zip(detections.tracker_id, detections.xyxy)):
            x_center = (bbox[0] + bbox[2]) / 2
            y_bottom = bbox[3]
            
            # Store position
            self.coordinates[tracker_id].append((x_center, y_bottom))
            
            # Determine lane position (left, middle, right)
            lane_pos = "middle"
            if x_center < frame_width/3:
                lane_pos = "left"
            elif x_center > 2*frame_width/3:
                lane_pos = "right"
            
            self.lane_positions[tracker_id].append(lane_pos)
            
            # Calculate speed if enough frames
            if len(self.coordinates[tracker_id]) >= self.fps/2:
                start = self.coordinates[tracker_id][0]
                end = self.coordinates[tracker_id][-1]
                distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                time = len(self.coordinates[tracker_id]) / self.fps
                speed = distance / time * 3.6  # Convert to km/h
                speeds[tracker_id] = speed
            
            # Detect lane changes
            if len(self.lane_positions[tracker_id]) >= 2:
                if self.lane_positions[tracker_id][-1] != self.lane_positions[tracker_id][-2]:
                    lane_changes[tracker_id] = True
                    
        return speeds, lane_changes

if __name__ == "__main__":
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize YOLO model
    model = YOLO("/Users/varun/Documents/yolo_weights/yolov8n.pt")
    model.to(device)
    
    # Initialize video capture
    video_path = "/Users/varun/Documents/major_additional_comps/vehicles.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Initialize vehicle tracker
    tracker = VehicleTracker(fps=fps)
    
    # Initialize ByteTrack
    byte_tracker = sv.ByteTrack()
    
    print("Processing video... Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Maintain aspect ratio while resizing
        scale = min(640/frame_width, 480/frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Detect vehicles
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter for vehicles (cars, trucks, buses)
        mask = np.array([class_id in [2, 5, 7] for class_id in detections.class_id])
        detections = detections[mask]
        
        # Apply tracking
        detections = byte_tracker.update_with_detections(detections)
        
        # Detect lanes
        lanes = detect_lanes(frame)
        if lanes is not None and len(lanes) > 0:
            for line in lanes:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Track vehicles and detect lane changes
        speeds, lane_changes = tracker.update(detections, new_width)
        
        # Prepare labels
        labels = []
        for tracker_id in detections.tracker_id:
            speed = speeds.get(tracker_id, 0)
            # Add color indicator in the label itself
            if speed > 100 or tracker_id in lane_changes:
                label = f" #{tracker_id} {int(speed)} km/h"  # Red circle for warning
            else:
                label = f" #{tracker_id} {int(speed)} km/h"  # Green circle for normal
            labels.append(label)
        
        # Draw annotations
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        # Display result
        cv2.imshow("Speed and Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()