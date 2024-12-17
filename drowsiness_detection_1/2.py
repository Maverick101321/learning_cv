import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

#Initializing the pyttsx3 engine for alert sound
engine = pyttsx3.init()

#Setting the camera to 1
camera = cv2.VideoCapture(1)

#Mapping the face for eye detection
face_detector = dlib.get_frontal_face_detector()

#Getting the .dat file for facial landmarks
dlib_face_landmark = dlib.shape_predictor("drowsiness_detection_1/shape_predictor_68_face_landmarks.dat")

#Calculating the eye aspect ratio
def detect_eye(eye):
    #Calculating the euclidean dist
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    aspect_ratio_eye = (a+b)/(2*c)
    return aspect_ratio_eye

#Main loop
#It will run until killed otherwise
while True:
    null, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detecting the faces
    faces = face_detector(gray)
    #Looping through the faces
    for face in faces:
        #Getting the facial landmarks
        face_landmarks = dlib_face_landmark(gray, face)
        left_eye = []
        right_eye = []
        #Points allocation for the left eyes in the .dat file is from 42 to 47
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            right_eye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        #Points allocation for the right eyes in the .dat file is from 36 to 41
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            left_eye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(9)
    if key == 20:
        break
camera.release()
camera.destroyAllWindows()
