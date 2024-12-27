#Here we're going to use the car vid
#And we're going to count the number of cars, motorbikes, buses and trucks coming in. 
#And instead of using cpu we're trying with mac's own mps(metal performance shaders)

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import torch
from sort import *

#Checking if mps is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

#capturing the video
cap = cv2.VideoCapture("../gfg_cv/vids_1_yolo/cars.mp4")

#loading the model and moving it to device
model = YOLO("../yolo_weights/yolov8l.pt").to(device)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
              "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

#defining the mask
mask = cv2.imread("counter/mask1.png")

#Tracking the objects
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#Line limits
limits = [257, 375, 673, 375]

#defining the total count var(x)
total_count = []

while True:
    ret, img = cap.read()
    if not ret:
        break

    #Applying the mask
    img_region = cv2.bitwise_and(img, mask)

    #Running inference directly on the raw frame
    #Assuming ultralytics will resize as needed
    res = model(img_region)

    #Getting the detections
    detections = np.empty((0, 5))

    for r in res:
        boxes = r.boxes
        for box in boxes:
            #Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            #Confidence
            #Calculting the confidence
            #Round the confidence to 2 decimal places
            conf = math.ceil((box.conf[0] * 100)) / 100

            #Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name == "car" or class_name == "motorbike" or class_name == "bus" \
                or class_name == "truck" and conf > 0.3:
            #Drawing the bounding box if the class is car (prev version)
                #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, 
                #                   thickness=1)
                #Appending the detections
                current_arr = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack((detections, current_arr))

    #Updating the tracker
    results_tracker = tracker.update(detections)
    #Drawing the line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for trackres in results_tracker:
        x1, y1, x2, y2, id = trackres
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(trackres)
        #Drawing the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #Drawing the id
        cvzone.putTextRect(img, f' {class_name} {int(id)}', (max(0, x1), max(35, y1)), scale=1, 
                           thickness=1)
        #Drawing the center
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        #Counting the vehicles
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            #Only counting if the count of that id in the total_count is 0 
            if total_count.count(id) == 0:
                total_count.append(id)
                #Displaying the line as green upon detection
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    #Displaying the total count
    cvzone.putTextRect(img, f' Total Count: {len(total_count)}', (50, 50))

    cv2.imshow("Result", img)
    #cv2.imshow("Img_region", img_region)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
