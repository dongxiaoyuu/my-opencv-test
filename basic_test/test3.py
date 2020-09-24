#  图像的绘制

import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)  # 绘制图像
print(img)

# img[:] = 100, 0, 0  # 上色(蓝色)
# print(img.shape)

cv2.line(img,(0,0),(300,300),(100,100,255),3)  # 绘制线条
cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)  # 绘制矩形
cv2.circle(img,(400,50),30,(255,255,255),5)  # 绘制圆
cv2.putText(img, "opencv",(300,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,200),1)  # 放置文本

cv2.imshow("Image", img)

cv2.waitKey(0)
