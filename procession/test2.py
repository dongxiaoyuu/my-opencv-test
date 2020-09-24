# 滤波,使用各种线性滤波器对图像进行平滑处理
import cv2

Img = cv2.imread('../Resources/noise.jpg')


cv2.imshow("image", Img)

# mean filtering
blur = cv2.blur(Img, (3, 3))
cv2.imshow("mean image", blur)

# box filtering
box = cv2.boxFilter(Img, -1, (3, 3), normalize=True)
cv2.imshow("box image", box)

# gaussianBlur filtering
gaussianBlur = cv2.GaussianBlur(Img, (5, 5), 1)
cv2.imshow("gaussian image", gaussianBlur)

# medianBlur filtering
medianBlur = cv2.medianBlur(Img, 5)
cv2.imshow("median image",medianBlur)
cv2.waitKey(0)