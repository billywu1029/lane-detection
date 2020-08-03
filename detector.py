import cv2
import numpy as np

GAUSSIAN_BLUR_KERNEL_SHAPE = (5, 5)  # Shape for Gaussian Blur kernel
GAUSSIAN_BLUR_STDDEV = 0  # Will apply for both x and y std devs

# Make sure these are in a 3:1 ratio, TODO: Cite some research/math to show the optimality of this
CANNY_LOW_THRESH = 50
CANNY_HI_THRESH = 150
ROI_TOP_WIDTH_COEFF = 0.4  # TODO: Need to analyze FOV and camera placement/height to show this encompasses current lane
ROI_TOP_HEIGHT_COEFF = 0.6  # TODO: To prove this correctness, we'd need guarantees about camera placement/height
ROI_MASK_FILL = 255

# Hough Transform Parameters
ACC_RHO = 2  # number of pixels of precision for each "bin" for the Hough 2D accumulator array/"grid" of bins
ACC_THETA = np.pi / 180  # number of radians of precision for each bin in Hough accumulator
BIN_THRESH = 100  # threshold for minimum number of "votes" for any Hough bin needed to accept the fit line
MIN_LANE_PIXEL_LEN = 40
MAX_LANE_PIXEL_GAP = 5  # TODO: Could be the solution for dashed lines!
LINE_DISPLAY_COLOR = (255, 0, 0)  # Blue lines
LINE_THICCNESS = 10

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
    return cv2.bitwise_and(mask, image)

def display_lines(image, lines):
    result = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # 1D array instead of 2D
            cv2.line(result, (x1, y1), (x2, y2), LINE_DISPLAY_COLOR, LINE_THICCNESS)
    return result


if __name__ == "__main__":
    img = cv2.imread('input.png')
    lane_img = np.copy(img)  # Need a copy to avoid aliasing issues mutating og image
    greyscale = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
    # 5x5 kernel for convolution; sigmaX = 0, default sigmaY = sigmaX -> std dev generated based on kernel
    blurred = cv2.GaussianBlur(greyscale, GAUSSIAN_BLUR_KERNEL_SHAPE, GAUSSIAN_BLUR_STDDEV)
    canny = cv2.Canny(blurred, CANNY_LOW_THRESH, CANNY_HI_THRESH)
    roi_canny = roi(canny)
    lines = cv2.HoughLinesP(roi_canny, ACC_RHO, ACC_THETA, BIN_THRESH, np.array([]), MIN_LANE_PIXEL_LEN, MAX_LANE_PIXEL_GAP)
    hough_img = display_lines(roi_canny, lines)
    print(lines)
    cv2.imshow("result", hough_img)
    cv2.waitKey(0)
    cv2.destroyWindow('result')
