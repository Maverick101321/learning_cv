from ultralytics import YOLO
import cv2
import os

#loading the model
model = YOLO("../yolo_weights/yolov8n.pt")
#defining the input img path
input_img_path = "experimenting_yolo/imgs/img_2.jpeg"

#processing the image
results = model(input_img_path)

#getting the output image
output_img = results[0].plot()
#defining the output image path
output_img_path = "experimenting_yolo/imgs/output_img_2.jpeg"
#saving the output img
cv2.imwrite(output_img_path, output_img)

#displaying the output img
cv2.imshow("Result", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()