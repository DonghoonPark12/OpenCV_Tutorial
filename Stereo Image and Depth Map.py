import numpy as np
import cv2
import matplotlib.pyplot as plt

imgL = cv2.imread('im0_motor.png', 0)
imgR = cv2.imread('im1_motor.png', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)

plt.xticks([]), plt.yticks([])
plt.imshow(disparity, 'Gray')
plt.show()