import cv2
from help import *

class EdgeFinder:
    def __init__(self, image, kernel_size=1,
                 low_threshold=50, high_threshold=150,
                 box_left_start_x=0, box_left_start_y=540,
                 box_left_end_x=430, box_left_end_y=330,
                 box_right_start_x=960, box_right_start_y=540,
                 box_right_end_x=510, box_right_end_y=330,
                 rho=2, theta=np.pi/180, threshold=15, min_line_len=40, max_line_gap=20):
        self.image = image
        self.img_dim = image.shape
        # gauss
        self._kernel_size = kernel_size
        # Canny
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        # Mask vertices :
        self._box_left_start_x = box_left_start_x
        self._box_left_start_y = box_left_start_y  # imshape[0]
        self._box_left_end_x = box_left_end_x
        self._box_left_end_y = box_left_end_y  # horizon
        self._box_right_start_x = box_right_start_x
        self._box_right_start_y = box_right_start_y  # imshape[0]
        self._box_right_end_x = box_right_end_x
        self._box_right_end_y = box_right_end_y  # imshape[0]
        # Hough Transform:
        self._rho = rho
        self._theta = theta
        self._ntheta = int(theta * 180 / np.pi)   # number of 1°
        self._threshold = threshold
        self._min_line_len = min_line_len
        self._max_line_gap = max_line_gap



        def on_kernel_size(pos):
            self._kernel_size = pos  # update the parameter value
            self._render()  # update the GUI

        def on_low_threshold(pos):
            self._low_threshold = pos  # update the parameter value
            self._render()  # update the GUI

        def on_high_threshold(pos):
            self._high_threshold = pos  # update the parameter value
            self._render()  # update the GUI

        def on_box_left_start_x(pos):
            self._box_left_start_x = pos  # update the parameter value
            self._render()  # update the GUI

        def on_box_left_start_y(pos):
            self._box_left_start_y = pos  # update the parameter value
            self._render()  # update the GUI

        def on_box_left_end_x(pos):
            self._box_left_end_x= pos  # update the parameter value
            self._render()  # update the GUI

        def on_box_left_end_y(pos):
            self._box_left_end_y = pos  # update the parameter value
            self._render()  # update the GUI

        def on_box_right_start_x(pos):
            self._box_right_start_x = pos  # update the parameter value
            self._render()  # update the GUI

        def on_box_right_start_y(pos):
            self._box_right_start_y = pos  # update the parameter value
            self._render()  # update the GUI

        def on_box_right_end_x(pos):
            self._box_right_end_x = pos  # update the parameter value
            self._render()  # update the GUI

        def on_box_right_end_y(pos):
            self._box_right_end_y = pos  # update the parameter value
            self._render()  # update the GUI

        # Hough Transform
        def on_rho(pos):
            self._nrho = pos  # update the parameter value
            self._rho = pos * np.pi/180
            self._render()  # update the GUI

        def on_theta(pos):
            self._theta = pos  # update the parameter value
            self._render()  # update the GUI

        def on_threshold(pos):
            self._threshold = pos  # update the parameter value
            self._render()  # update the GUI

        def on_min_line_len(pos):
            self._min_line_len = pos  # update the parameter value
            self._render()  # update the GUI

        def on_max_line_gap(pos):
            self._max_line_gap = pos  # update the parameter value
            self._render()  # update the GUI

        cv2.namedWindow('mask')
        cv2.namedWindow('gauss')
        cv2.namedWindow('canny')
        cv2.namedWindow('hough_segments')

        # make track bars for the parameters
        cv2.createTrackbar('gauss_kernel', 'gauss', self._kernel_size, 25,
                           on_kernel_size)
        cv2.createTrackbar('canny_low_threshold', 'canny', self._low_threshold, 255,
                           on_low_threshold)
        cv2.createTrackbar('canny_high_threshold', 'canny', self._high_threshold, 255,
                           on_high_threshold)
        # region masking
        cv2.createTrackbar('mask_left_start_x', 'mask', self._box_left_start_x, 960,
                           on_box_left_start_x)
        cv2.createTrackbar('mask_left_start_y', 'mask', self._box_left_start_y, 540,
                           on_box_left_start_y)
        cv2.createTrackbar('mask_left_end_x', 'mask', self._box_left_end_x, 960,
                           on_box_left_end_x)
        cv2.createTrackbar('mask_left_end_y', 'mask', self._box_left_end_y, 540,
                           on_box_left_end_y)
        cv2.createTrackbar('mask_right_start_x', 'mask',
                           self._box_right_start_x, 960,
                           on_box_right_start_x)
        cv2.createTrackbar('mask_right_start_y', 'mask',
                           self._box_right_start_y, 540,
                           on_box_right_start_y)
        cv2.createTrackbar('mask_right_end_x', 'mask', self._box_right_end_x, 960,
                           on_box_right_end_x)
        cv2.createTrackbar('mask_right_end_y', 'mask', self._box_right_end_y, 540,
                           on_box_right_end_y)

        # Hough Transform
        cv2.createTrackbar('rho', 'hough_segments', self._rho, 10,
                           on_rho)
        cv2.createTrackbar('theta', 'hough_segments', self._ntheta, 10,
                           on_theta)
        cv2.createTrackbar('threshold', 'hough_segments', self._threshold, 100,
                           on_threshold)
        cv2.createTrackbar('min_line_len', 'hough_segments', self._min_line_len, 100,
                           on_min_line_len)
        cv2.createTrackbar('max_line_gap', 'hough_segments', self._max_line_gap, 100,
                           on_max_line_gap)


        self._render()

        print ("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def _render(self):
        # gray scale image
        self._gray_img = grayscale(self.image)
        #cv2.imshow('gauss', self._gauss_img)
        # apply gauss
        self._gauss_img = gaussian_blur(self._gray_img, self._kernel_size)
        cv2.imshow('gauss', self._gauss_img)

        # apply Canny Edge Detection
        self._edges = canny(self._gauss_img, self._low_threshold,
                            self._high_threshold)
        cv2.imshow('canny', self._gauss_img)

        # mask the region we are not interested in
        vertices = np.array([[(self._box_left_start_x, self._box_left_start_y),
                              (self._box_left_end_x, self._box_left_end_y),
                              (self._box_right_end_x, self._box_right_end_y),
                              (self._box_right_start_x,
                               self._box_right_start_y)]],
                            dtype=np.int32)
        self._masked_edges = region_of_interest(self._edges, vertices=vertices)
        cv2.imshow('mask', self._masked_edges)

        # apply Hough to find segments in the detected edges
        self._line_segments_img = hough_lines(self._masked_edges, self._rho
                                              , self._theta, self._threshold,
                                              self._min_line_len,
                                              self._max_line_gap,
                                              segmented=True)
        self._image_with_segments = weighted_img(self._line_segments_img,
                                                 initial_img=self.image)

        cv2.imshow('hough_segments', self._image_with_segments)

        # show final result