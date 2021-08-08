# This programs calculates the orientation of an object.
# The input is an image, and the output is an annotated image
# with the angle of otientation for each object (0 to 180 degrees)
import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np

erosion_size = 0
max_elem = 2
max_kernel_size = 21

# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

def erode_img(mode, erosion_size, src):
    #erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_shape = morph_shape(mode)
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    eroded_img = cv2.erode(src, element)
    return eroded_img

def binarize_img(src, threshold):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    _, binarized_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binarized_img

def detect_rot(img, erosion=False):
    angle = 0
    # Was the image there?
    if img is None:
        print("Error: File not found")
        exit(0)

    # Convert image to grayscale
    bw = binarize_img(img, 245)
    if erosion:
        bw = erode_img(0, 10, bw)

    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 3700 or 100000 < area:
            continue

        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        #print("--->{}".format(int(rect[2])))
        angle = int(rect[2])

        if width > height:
            angle = 90 - angle
        else:
            angle = -angle

    if -91 < angle < 0:
        print("Detection completed: {} degrees".format(str(90 + angle)))
        return 90 + angle
    elif angle >=0:
        print("++++++++++++++++++Detection completed: {} degrees".format(str(angle)))
        return angle
    else:
        return -angle