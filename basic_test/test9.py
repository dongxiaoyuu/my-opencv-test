# 通道分离、合并
import cv2
scr = cv2.imread("../Resources/gakki.jpg")
b,g,r = cv2.split(scr)
cv2.imshow("blue", b)
cv2.imshow("red", r)
cv2.imshow("green", g)

scr[:,2,:] = 0
cv2.imshow("change img", scr)
scr2 = cv2.merge([b,g,r])
cv2.imshow("change2 img", scr2)
cv2.waitKey(0)


