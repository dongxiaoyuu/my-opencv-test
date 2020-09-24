# 直方图
# 通过查看图像的直方图，您可以直观地了解该图像的对比度，亮度，强度分布等

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('../Resources/haer.jpg',0)
cv.imshow("img",img)
plt.hist(img.ravel(),256,[0,256]); plt.show()

mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(img,img,mask = mask)
# 计算掩码区域和非掩码区域的直方图
# 检查作为掩码的第三个参数
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()