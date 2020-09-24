# 轮廓
import numpy as np
import cv2 as cv
im = cv.imread('../Resources/haer.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cimg = cv.drawContours(im, contours, -1, (0,0,255), 1)
cv.imshow("contours",cimg)#  在图像中绘制所有轮廓
c2img = cv.drawContours(im, contours, 3, (0,255,0), 3)
cv.imshow("contours2",c2img)

cv.waitKey(0)