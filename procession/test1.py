# OpenCV中的阈值(threshold)函数： threshold 的运用。这个函数有5种阈值化类型

import cv2
import matplotlib.pyplot as plt

Img = cv2.imread('../Resources/haer.jpg')

# > 127 is 255
ret1, threshold1 = cv2.threshold(Img, 127, 255, cv2.THRESH_BINARY)
# > 127 is 255 inverse
ret2, threshold2 = cv2.threshold(Img, 127, 255, cv2.THRESH_BINARY_INV)
# > 127 is 127
ret3, threshold3 = cv2.threshold(Img, 127, 255, cv2.THRESH_TRUNC)
# < 127 is 0
ret4, threshold4 = cv2.threshold(Img, 127, 255, cv2.THRESH_TOZERO)
# < 127 is 0 inverse
ret5, threshold5 = cv2.threshold(Img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original', 'Binary', 'BinaryInv', 'Trunc', 'ToZero', 'ToZeroInv']
images = [Img, threshold1, threshold2, threshold3, threshold4, threshold5]

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    # plt.xticks([])
    # plt.yticks([])

plt.show()