import cv2
import numpy as np

#taking input from webcam
vid = cv2.VideoCapture(0)

#now running a while loop until the letter 'q' is pressed
while True:
    #capturing the current frame
    _, frame = vid.read()

    #displaying the current frame
    cv2.imshow("Frame", frame)

    #setting the values for base colors

    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]

    #calculating the mean of the base colors
    blue_mean = np.mean(blue)
    green_mean = np.mean(green)
    red_mean = np.mean(red)

    #displaying or printing the most prominent color
    if blue_mean > green_mean and blue_mean > red_mean:
        print("Blue")
    elif green_mean > blue_mean and green_mean > red_mean:
        print("Green")
    else:
        print("Red")
    
    #breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releasing the webcam and closing the window
vid.release()
cv2.destroyAllWindows()
