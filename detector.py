import cv2
import numpy as np

img = cv2.imread('input.png')
lane_img = np.copy(img)  # Need a copy to avoid aliasing issues mutating og image
greyscale = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)

