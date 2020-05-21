"""
How to run:
python find_edges.py <image path>
"""

import argparse
import cv2
import os

from guiutils import EdgeFinder


def main():
    # parser for the command line input (automaticlly generates help)
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')
    # save the arguments
    args = parser.parse_args()
    # read the image given as an argument
    img = cv2.imread(args.filename)
    # create the edge finding and gui class
    edge_finder = EdgeFinder(img, kernel_size=5)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()