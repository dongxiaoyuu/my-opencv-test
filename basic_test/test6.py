#  改变HSV色彩空间
import cv2


img = cv2.imread("../Resources/haer.jpg")
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

cv2.imshow("Original", img)
cv2.imshow("HSV", imgHSV)
cv2.waitKey(0)