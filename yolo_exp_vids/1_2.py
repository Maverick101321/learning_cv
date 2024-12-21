#Here we're going to use vids
#Also we're going to try to use the yolov8 large model 
#And instead of using cpu we're trying with mac's own mps(metal performance shaders)

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import torch

#Checking if mps is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(device)

#capturing the video
cap = cv2.VideoCapture("../gfg_cv/vids_1_yolo/bikes.mp4")

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

while True:
    ret, img = cap.read()
    if not ret:
        break

    #Resize the image to a compatible shape
    img_resized = cv2.resize(img, (640, 640))

    #normalize the image
    img_resized = img_resized / 255.0

    #Converting the image to a tensor and moving it to device
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    #Running the model on the image tensor
    res = model(img_tensor, stream=True)
    for r in res:
        boxes = r.boxes
        for box in boxes:
            #Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #Confidence
            #Calculting the confidence
            #Round the confidence to 2 decimal places
            conf = math.ceil((box.conf[0] * 100)) / 100

            #Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)))

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
