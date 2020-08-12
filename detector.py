import cv2
import numpy as np

GAUSSIAN_BLUR_KERNEL_SHAPE = (5, 5)  # Shape for Gaussian Blur kernel
GAUSSIAN_BLUR_STDDEV = 0  # Will apply for both x and y std devs, 0 to let stddev be generated by kernel

# Canny constants
# Make sure these are in a 3:1 ratio, TODO: Cite some research/math to show the optimality of this
CANNY_LOW_THRESH = 50
CANNY_HI_THRESH = 150
ROI_TOP_WIDTH_COEFF = 0.4  # TODO: Need to analyze FOV and camera placement/height to show this encompasses current lane
ROI_TOP_HEIGHT_COEFF = 0.6  # TODO: To prove this correctness, we'd need guarantees about camera placement/height
ROI_MASK_FILL = 255

# Averaging lane constants
LANE_HEIGHT_COEFF = ROI_TOP_HEIGHT_COEFF  # Determines how far the overlaid lane stretches on the og image
LINE_POLY_ORDER = 1  # Order of the polynomial describing the lane line

# Hough Transform constants
ACC_RHO = 2  # number of pixels of precision for each "bin" for the Hough 2D accumulator array/"grid" of bins
ACC_THETA = np.pi / 180  # number of radians of precision for each bin in Hough accumulator
BIN_THRESH = 100  # threshold for minimum number of "votes" for any Hough bin needed to accept the fit line
MIN_LANE_PIXEL_LEN = 10
MAX_LANE_PIXEL_GAP = 100  # TODO: Could be the solution for dashed lines!
LINE_DISPLAY_COLOR = (255, 0, 0)  # Blue lines
LINE_THICCNESS = 5
OG_IMG_OVERLAY_WEIGHT = 0.8
HOUGH_IMG_OVERLAY_WEIGHT = 1
OVERLAY_GAMMA = 1  # Throwaway value to add to final weighted sum in addWeighted()

def canny(image):
    greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(greyscale, GAUSSIAN_BLUR_KERNEL_SHAPE, GAUSSIAN_BLUR_STDDEV)
    return cv2.Canny(blurred, CANNY_LOW_THRESH, CANNY_HI_THRESH)

def roi(image):
    h, w = image.shape[0], image.shape[1]
    # b_left = (0, h)
    # b_right = (w, h)
    # t_left = (ROI_TOP_WIDTH_COEFF * w, ROI_TOP_HEIGHT_COEFF * h)
    # t_right = ((1 - ROI_TOP_WIDTH_COEFF) * w, ROI_TOP_HEIGHT_COEFF * h)
    # TODO: Dashcam vs standard self-driving camera placement roi bounding box points
    b_left = (0, h * 0.9)
    b_right = (w, h * 0.9)
    t_left = (0.3 * w, 0.47 * h)
    t_right = (0.7 * w, 0.47 * h)
    points = np.array([[b_left, t_left, t_right, b_right]])
    points = np.int32(points)  # Fix opencv bug requiring cast to 32-bit ints
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, points, ROI_MASK_FILL)
    return cv2.bitwise_and(mask, image)

def cvt_slopeint_coords(h, params):
    slope, intercept = params[0], params[1]
    y1 = h  # Start from bottom of image
    y2 = int(y1 * LANE_HEIGHT_COEFF)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def avg_fit_lanes(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), LINE_POLY_ORDER)
        slope, intercept = params[0], params[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:  # Vertical lines seem to end up here, TODO: what if vertical poles end up in FOV/ROI
            right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line_coords = cvt_slopeint_coords(image.shape[0], left_fit_avg)
    right_line_coords = cvt_slopeint_coords(image.shape[0], right_fit_avg)
    return np.array([left_line_coords, right_line_coords])

def display_lines(image, lines):
    result = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # unpack coords from 2D array
            cv2.line(result, (x1, y1), (x2, y2), LINE_DISPLAY_COLOR, LINE_THICCNESS)
    return result

def video_lane_overlay(filename):
    """Takes in a dashcam video of driving on a straight road and overlays detected lane lines."""
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        _, frame = video.read()
        line_overlay = img_lane_detect(frame)
        cv2.imshow("lane detection frame", line_overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def img_lane_detect(img):
    lane_img = np.copy(img)  # Need a copy to avoid aliasing issues mutating og image
    canny_img = canny(lane_img)
    roi_canny = roi(canny_img)
    hough_lines = cv2.HoughLinesP(roi_canny, ACC_RHO, ACC_THETA, BIN_THRESH, np.array([]), MIN_LANE_PIXEL_LEN,
                                  MAX_LANE_PIXEL_GAP)
    # avg_lines = avg_fit_lanes(lane_img, hough_lines)
    # hough_img = display_lines(lane_img, avg_lines)
    hough_img = display_lines(lane_img, hough_lines)
    line_overlay = cv2.addWeighted(lane_img, OG_IMG_OVERLAY_WEIGHT, hough_img, HOUGH_IMG_OVERLAY_WEIGHT,
                                   OVERLAY_GAMMA)
    return line_overlay

if __name__ == "__main__":
    # Video borrowed from the Comma2K dataset:
    # https://github.com/commaai/comma2k19/blob/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/video.hevc
    v_file = "video.hevc"
    video_lane_overlay(v_file)

