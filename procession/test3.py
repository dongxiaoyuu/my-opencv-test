# 使用OpenCV函数 morphologyEx 进行形态学操作：
#
#     开运算 (Opening)
#     闭运算 (Closing)
#     形态梯度 (Morphological Gradient)
#     顶帽 (Top Hat)
#     黑帽(Black Hat)

import cv2
import numpy as np

def showImg(winname, mat):
    cv2.imshow(winname, mat)
    cv2.waitKey(0)


img = cv2.imread('../Resources/heidi.jpg')
# 开运算闭运算
kernel = np.ones((3, 3), np.uint8)
showImg("image",img)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
showImg('opening', opening)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
showImg('closing', closing)
#  梯度
kernel = np.ones((7, 7), np.uint8)

dilatePie = cv2.dilate(img, kernel, iterations=5)
erodePie = cv2.erode(img, kernel, iterations=5)
dilateErode = np.hstack((dilatePie, erodePie))
showImg('dilateErode', dilateErode)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
showImg('gradient', gradient)
# 礼帽
# 黑帽
kernel = np.ones((3, 3), np.uint8)

topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
showImg('topHat', topHat)
showImg('blackHat', blackHat)