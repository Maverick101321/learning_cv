from ultralytics import YOLO
import cv2
import cvzone

#capturing the video
cap = cv2.VideoCapture(0)

#setting the width and height of the video
cap.set(3, 1280)
cap.set(4, 720)

#loading the model
model = YOLO("../yolo_weights/yolov8n.pt")

while True:
    null, img = cap.read()
    res = model(img, stream=True)
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            #Checking if we're getting the correct bounding boxes
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()