import cv2
import numpy as np

img = cv2.imread('input.png')
lane_img = np.copy(img)  # Need a copy to avoid aliasing issues mutating og image
greyscale = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
# 5x5 kernel for convolution; sigmaX = 0, default sigmaY = sigmaX -> std dev generated based on kernel
blurred = cv2.GaussianBlur(greyscale, (5, 5), 0)

