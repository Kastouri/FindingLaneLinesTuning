import cv2
import numpy as np
from numpy.polynomial.polynomial import polyfit

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_line_segments(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=5,
               start_left_x=0, start_left_y=540,
               end_left_x=450, end_left_y=325,
               start_right_x=960, start_right_y=540, end_right_x=510, end_right_y=325,
               ):
    # initiate point list for right and left lane
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []
    # initiate slope
    slope = 1
    # iterate through lines
    for line in lines:
        for x1, y1, x2, y2 in line:  # for points in line
            # calculate slope
            slope = (y1 - y2) / (x1 - x2)
            # add points to corresponding lane
            if slope > 0:  # and (((x1+x2)/2) < (img.shape[1] / 2)) : # if slope is positive to the left
                left_lane_x.extend([x1, x2])
                left_lane_y.extend([y1, y2])
            elif slope < 0:  # and (((x1+x2)/2) > (img.shape[1] / 2)):  # if slope is negative to the right
                right_lane_x.extend([x1, x2])
                right_lane_y.extend([y1, y2])

    #global kernel_size, low_threshold, high_threshold, box_left_start_x, box_left_start_y, box_left_end_x, box_left_end_y
    #global box_right_start_x, box_right_end_x, box_right_start_y, box_right_end_y
    #global rho, theta, threshold, min_line_len, max_line_gap
    #global vertices

    if (right_lane_x) and (left_lane_y):  # if there is something to draw
        # determine equation desscribing the lane line
        b_right, m_right = polyfit(np.array(right_lane_x), np.array(right_lane_y), 1)
        b_left, m_left = polyfit(np.array(left_lane_x), np.array(left_lane_y), 1)

        # determine x-position of start point given y = shape[0]
        start_right_x = int(round((start_right_y - b_right) /
                                  m_right)) if m_right != 0 else 860
        start_left_x = int(round((start_left_y - b_left) /
                                 m_left)) if m_left != 0 else 100

        # ending point
        end_right_x = int(round((end_right_y - b_right) /
                                m_right)) if m_right != 0 else end_right_x
        end_left_x = int(round((end_left_y - b_left) /
                               m_left)) if m_left != 0 else end_left_x

        # draw lines
        cv2.line(img, (start_left_x, start_left_y),
                 (end_left_x, end_left_y), color, thickness)
        cv2.line(img, (start_right_x, start_right_y),
                 (end_right_x, end_right_y), color, thickness)

    return


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, segmented=False):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if segmented:
        draw_line_segments(line_img, lines)
    else:
        draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=1, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)