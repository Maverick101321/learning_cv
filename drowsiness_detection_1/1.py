import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

#Initializing the pyttsx3 engine for alert sound
engine = pyttsx3.init()

#Setting the camera to 1
camera = cv2.VideoCapture(1)

#Main loop
#It will run until killed otherwise
while True:
    #Reading a frame from the camera
    null, frame = camera.read()
    #Converting the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(9) 
    if key == 20:
        break
camera.release()
camera.destroyAllWindows()