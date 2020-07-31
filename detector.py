import cv2
import numpy as np

# Make sure these are in a 3:1 ratio, TODO: Cite some research/math to show the optimality of this
CANNY_LOW_THRESH = 50
CANNY_HI_THRESH = 150

img = cv2.imread('input.png')
lane_img = np.copy(img)  # Need a copy to avoid aliasing issues mutating og image
greyscale = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
# 5x5 kernel for convolution; sigmaX = 0, default sigmaY = sigmaX -> std dev generated based on kernel
blurred = cv2.GaussianBlur(greyscale, (5, 5), 0)
canny = cv2.Canny(blurred, CANNY_LOW_THRESH, CANNY_HI_THRESH)

