import argparse
from collections import defaultdict, deque
from sys import flags

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import supervision as sv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = YOLO("/Users/varun/Documents/yolo_weights/yolov8n.pt")
    model.to(device)

    box_annotator = sv.BoxAnnotator(thickness=4)
    
    # Open video capture
    cap = cv2.VideoCapture(args.source_video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        cv2.imshow("Vehicle Speed Estimation", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
