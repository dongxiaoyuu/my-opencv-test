#  图像的大小

import cv2


img = cv2.imread("../Resources/haer.jpg")
print(img.shape)

imgReSize = cv2.resize(img, (200, 300))  # 缩放大小
imgCropped = img[0:200,200:500]  # 图像剪裁

cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgReSize)
cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)