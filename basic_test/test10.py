import cv2


img = cv2.imread("../Resources/haer.jpg")
# ROI操作
face = img[50:250, 100:300]
gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
backface = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
img[50:250, 100:300] = backface
cv2.imshow("face", face)

cv2.imshow("Image", img)
cv2.waitKey(0)
