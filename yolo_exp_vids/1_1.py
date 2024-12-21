#Same as the pevios one but with fancy rectangles. 
#The previos file was using opencv and this one is the same using cvzone

from ultralytics import YOLO
import cv2
import cvzone
import math

#capturing the video from webcam
# cap = cv2.VideoCapture(0)
#setting the width and height of the video
#cap.set(3, 1280)
#cap.set(4, 720)

#for getting vid
cap = cv2.VideoCapture("../gfg_cv/vids_1_yolo/bikes.mp4")

#loading the model
model = YOLO("../yolo_weights/yolov8n.pt")

#For saving the output video
#Defining the codec and creating a VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))

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
    null, img = cap.read()
    res = model(img, stream=True)
    for r in res:
        boxes = r.boxes
        for box in boxes:
            #Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            #Confidence
            #Calculting the confidence
            #Round the confidence to 2 decimal places
            conf = math.ceil((box.conf[0] * 100)) / 100

            #Class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)))

    #Write the frame into the file 'output.avi'
    #out.write(img)    

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
#out.release()
cv2.destroyAllWindows()