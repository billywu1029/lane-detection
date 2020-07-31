import cv2
import numpy as np

# Make sure these are in a 3:1 ratio, TODO: Cite some research/math to show the optimality of this
CANNY_LOW_THRESH = 50
CANNY_HI_THRESH = 150
ROI_TOP_WIDTH_COEFF = 0.4  # TODO: Need to analyze FOV and camera placement/height to show this encompasses current lane
ROI_TOP_HEIGHT_COEFF = 0.5  # TODO: To prove this correctness, we'd need guarantees about camera placement/height
ROI_MASK_FILL = 255

def roi(image):
    h, w = image.shape[0], image.shape[1]
    b_left = (0, h)
    b_right = (w, h)
    t_left = (ROI_TOP_WIDTH_COEFF * w, ROI_TOP_HEIGHT_COEFF * h)
    t_right = ((1 - ROI_TOP_WIDTH_COEFF) * w, ROI_TOP_HEIGHT_COEFF * h)
    points = np.array([[b_left, t_left, t_right, b_right]])
    points = np.int32(points)  # Fix opencv bug requiring cast to 32-bit ints
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, points, ROI_MASK_FILL)
    return mask


img = cv2.imread('input.png')
lane_img = np.copy(img)  # Need a copy to avoid aliasing issues mutating og image
greyscale = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
# 5x5 kernel for convolution; sigmaX = 0, default sigmaY = sigmaX -> std dev generated based on kernel
blurred = cv2.GaussianBlur(greyscale, (5, 5), 0)
canny = cv2.Canny(blurred, CANNY_LOW_THRESH, CANNY_HI_THRESH)
cv2.imshow("result", roi(canny))
cv2.waitKey(0)

