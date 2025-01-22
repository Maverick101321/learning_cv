#only detecting the card as of right now. 
#using the trained model to detect the cards
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import torch
import pokerhandfunc as phf

#Checking if mps is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

#Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#loading the model and moving it to device
model = YOLO("../gfg_cv/Poker_Hand_Detector/best.pt").to(device)

classNames = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', 
              '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 
              'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']

while True:
    ret, img = cap.read()
    if not ret:
        break

    #Running inference directly on the raw frame
    #Assuming ultralytics will resize as needed
    res = model(img)
    hand = []

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

            #Drawing the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            if conf > 0.3:
                hand.append(class_name) # Append the class name to the hand list
    
    hand = list(set(hand))
    print(hand)

    if len(hand) == 5:
        res = phf.find_poker_hand(hand)
        print(res)
        cvzone.putTextRect(img, f'Your Hand: {res}', (300, 75), scale=3, thickness=5)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
