import cv2
import numpy as np

img = cv2.imread("../Resources/haer.jpg")

imgHor = np.hstack ((img,img))  # 图片水平堆叠
imgVar = np.vstack((img, img))  # 垂直堆叠

cv2.imshow("HImage", imgHor)
cv2.imshow("VImage", imgVar)
cv2.waitKey(0)