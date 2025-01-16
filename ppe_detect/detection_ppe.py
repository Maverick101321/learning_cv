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
print("Using device:", device)

#capturing the video
cap = cv2.VideoCapture("../gfg_cv/vids_1_yolo/ppe-3-1.mp4")

#loading the model and moving it to device
model = YOLO("../gfg_cv/ppe_detect/codes/runs/detect/train/weights/best.pt").to(device)

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 
              'Safety Vest', 'machinery', 'vehicle']

Color_sel = (0, 0, 255)

while True:
    ret, img = cap.read()
    if not ret:
        break

    #Running inference directly on the raw frame
    #Assuming ultralytics will resize as needed
    res = model(img)

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
            current_class = classNames[cls]

            if conf>0.5:
                #If class is hardhat, safety vest or mask then green colour
                if current_class == 'Hardhat' or current_class == 'Mask' or current_class == 'Safety Vest':
                    Color_sel = (0, 255, 0)# Green
                elif current_class == 'NO-Hardhat' or current_class == 'NO-Mask' or current_class == 'NO-Safety Vest':
                    Color_sel = (0, 0, 255)# Red
                else:
                    Color_sel = (255, 0, 0)

                #Drawing the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), Color_sel, 2)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, 
                               colorB=Color_sel, colorT=(255, 255, 255), colorR=Color_sel)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
