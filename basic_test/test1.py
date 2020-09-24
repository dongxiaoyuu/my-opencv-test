#  图像的灰度

import cv2
import numpy as np

img = cv2.imread("../Resources/haer.jpg")
kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 设置灰色图像
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)  # 模糊图像
imgCanny = cv2.Canny(img, 100, 100)  # 边缘检测
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)  # 图像扩张（膨胀）
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)  # 图像侵蚀
cv2.imshow("Image", img)
cv2.imshow("Gray Image", imgGray)
cv2.imshow("Blur Image", imgBlur)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dialation Image", imgDialation)
cv2.imshow("Eroded Image", imgEroded)
cv2.waitKey(0)
