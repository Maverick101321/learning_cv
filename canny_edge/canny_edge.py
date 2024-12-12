import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#Defining the canny edge detection function
#Below weak_th and strong_th are the thresholds for the double thresholding step

def canny_edge_detector(img, weak_th=None, strong_th=None):
    #Converting img to greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)

    #Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    #Converting cartesian coordinates to polar coordinates
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    #Setting the minimum and the maximum thresholds for double thresholding
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5

    #Getting the dimensions of the input image
    height, width = img.shape

    #Looping through the image
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)

            #Selecting the neighbours of the target pixel
            #By using the gradient direction
            #In x axis direction
            if grad_ang <= 22.5:
                neigh_1_x, neigh_1_y = i_x - 1, i_y
                neigh_2_x, neigh_2_y = i_x + 1, i_y
            
            #In top-right direction
            elif 22.5 < grad_ang <=(22.5 + 45):
                neigh_1_x, neigh_1_y = i_x - 1, i_y - 1
                neigh_2_x, neigh_2_y = i_x + 1, i_y + 1

            #In y axis direction
            elif (22.5 + 45) < grad_ang <= (22.5 + 90):
                neigh_1_x, neigh_1_y = i_x, i_y - 1
                neigh_2_x, neigh_2_y = i_x, i_y + 1

            #In top-left direction
            elif (22.5 + 90) < grad_ang <= (22.5 + 135):
                neigh_1_x, neigh_1_y = i_x - 1, i_y + 1
                neigh_2_x, neigh_2_y = i_x + 1, i_y - 1

            #Now restarting the cycle
            elif (22.5 + 135) < grad_ang <= 180:
                neigh_1_x, neigh_1_y = i_x - 1, i_y
                neigh_2_x, neigh_2_y = i_x + 1, i_y

            #Non-maximum supression
            if width > neigh_1_x >= 0 and height > neigh_1_y >= 0:
                if mag[i_y, i_x] < mag[neigh_1_y, neigh_1_x]:
                    mag[i_y, i_x] = 0
                    continue
            if width > neigh_2_x >= 0 and height > neigh_2_y >= 0:
                if mag[i_y, i_x] < mag[neigh_2_y, neigh_2_x]:
                    mag[i_y, i_x] = 0

    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)
    ids = np.zeros_like(img)

    #Double thresholding step
    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >=weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2
    
    #returning the magnitude of the gradients of edges
    return mag

frame = cv2.imread("img.jpeg")
canny_img = canny_edge_detector(frame)

#Displaying the input and the output images
plt.figure()
f, plots = plt.subplots(2, 1)  
plots[0].imshow(frame) 
plots[1].imshow(canny_img) 