# 选择部分图像
import cv2
import numpy as np

width,height = 250,350
img = cv2.imread("../Resources/haer.jpg")
pts1 = np.float32([[100,100],[100,200],[200,100],[200,200]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutPut = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("Image", img)
cv2.imshow("Output", imgOutPut)
cv2.waitKey(0)