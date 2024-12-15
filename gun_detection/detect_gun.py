import numpy as np
import cv2
import imutils
import datetime
import matplotlib.pyplot as plt

#Loading the XML classifier
#This xml file is already pre-trained to detect guns
gun_cascade = cv2.CascadeClassifier('gun_detection/cascade.xml')
#Loading the camera
camera = cv2.VideoCapture(0)
first_frame = None
gun_exist = False

#Looping through the camera frames
while True:
    #Reading the camera frame
    ret, frame = camera.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    #Converting the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detecting the guns
    gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    if len(gun) > 0:
        gun_exist = True
    #Now as guns are present, it will return the positions of the detected guns as Rect(x,y,w,h)
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #Extracting roi from grayscale and color frames
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    if first_frame is None:
        first_frame = gray
        continue
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    if gun_exist:
        print("Gun Detected")
        plt.imshow(frame)
        break
    else:
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
camera.release()
cv2.destroyAllWindows()

