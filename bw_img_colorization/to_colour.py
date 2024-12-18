import numpy as np
import cv2
from cv2 import dnn

#Model file paths
#Model arch
proto_file = 'bw_img_colorization/Model/colorization_deploy_v2.prototxt'
#Pretrained weights
model_file = 'bw_img_colorization/Model/colorization_release_v2.caffemodel'
#Cluster points
hull_pnts = 'bw_img_colorization/Model/pts_in_hull.npy'
img_path = 'bw_img_colorization/img.jpeg'

#Read the model params
#Loading the model
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pnts)

#Reading and preprocessing the image
img = cv2.imread(img_path)
scaled_img = img.astype('float32') / 255.0
#Converting from bgr to lab color space
#This will separate the luminance(l) and chrominance(ab) values
lab_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2Lab)

#Adding the cluster centres as 1x1 convolutions to the model
#Model is configured to use the cluster centres for colorization
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
#reshaping cluster points
pts = kernel.transpose().reshape(2, 313, 1, 1)
#Assigning cluster points to the model layers
net.getLayer(class8).blobs = [pts.astype('float32')]
net.getLayer(conv8).blobs = [np.full((1, 313), 2.606, np.float32)]

#Resizing lab img to 224x224, expected by the model
resized = cv2.resize(lab_img, (224, 224))
#Splitting the L channel
L = cv2.split(resized)[0]
#Mean subtraction to normilize the input
L -= 50

#Predicting the ab channels from the L channel
#Setting the L channel as input
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
#Resizing the ab channels to the original image size
ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

#Taking the L channel from the image
L = cv2.split(lab_img)[0]
#Combining the L channel with the predicted ab channels
colorized_img = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

#Converting the image from lab to bgr
colorized_img = cv2.cvtColor(colorized_img, cv2.COLOR_Lab2BGR)
colorized_img = np.clip(colorized_img, 0, 1)

#Change the image to 0-255 range and convert to uint8
colorized_img = (255 * colorized_img).astype('uint8')

#Saving the output img in the current directory
output_img_path = 'colorized_img.jpeg'
cv2.imwrite(output_img_path, colorized_img)
