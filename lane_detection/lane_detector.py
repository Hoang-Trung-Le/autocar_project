import math

import cv2
import imutils
import numpy as np


def interest_area(w, h, padding=False):
    return np.array([[(0, h * 0.95), (w / 2, h * 0.66 + (2 if padding else 0)), (w, h * 0.95)]], np.int)


def mask_interest_area(img, padding=False):
    h, w = img.shape[:2]
    interest_mask = interest_area(w, h, padding)
    stencil = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(stencil, interest_mask, (255, 255, 255))
    return cv2.bitwise_and(img, stencil)


def process_input(img):
    w = 1024
    h = int(img.shape[0] * w / img.shape[1])
    img = cv2.resize(img, (w, h))
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray, mask_interest_area(img), mask_interest_area(img_gray)


def apply_color_threshold(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    s_binary = np.zeros_like(s)
    s_binary[(s >= 170) & (s <= 255)] = 255
    return s_binary


def get_edge_image(img, img_gray):
    s_binary = apply_color_threshold(img)
    canny = cv2.Canny(cv2.GaussianBlur(img_gray, (7, 7), 0), 100, 200)
    combined_img = np.zeros_like(s_binary)
    combined_img[(s_binary == 255) | (canny == 255)] = 255
    return mask_interest_area(combined_img, padding=True)


def line_length(x1, y1, x2, y2):
    return abs(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def intersection(line1, line2):
    x_diff = (line1[0] - line1[2], line2[0] - line2[2])
    y_diff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        return None

    d = (
        det(*[[line1[0], line1[1]], [line1[2], line1[3]]]),
        det(*[[line2[0], line2[1]], [line2[2], line2[3]]]))
    x = int(det(d, x_diff) / div)
    y = int(det(d, y_diff) / div)
    return [x, y]


def detect_lane_on_side(lines, w, h, side='right'):
    lane = []
    min_distance_to_center = w

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            if (side == 'right' and 30 < angle < 80) or (side == 'left' and -80 < angle < -20):
                distance = line_length(min(x2, x1), max(y1, y2), w / 2, h)
                if distance < min_distance_to_center:
                    lane = line[0]
                    min_distance_to_center = distance
                    # print(side, distance, angle, x1, x2, y1, y2)

    if len(lane) == 4:
        # Extend line to fit image size
        intersect_lower_bound = intersection(lane, [0, h, w, h])
        intersect_upper_bound = intersection(lane, [0, h * 0, w, h * 0])
        lane = [intersect_lower_bound[0], h, intersect_upper_bound[0], 0]

    return lane


def debug(img, mask):
    debug_img = img.copy()
    overlay = np.zeros_like(debug_img)
    cv2.fillPoly(overlay, mask, (0, 255, 0))
    debug_img = cv2.addWeighted(debug_img, 1, overlay, 0.3, 0)
    cv2.imshow('debug', debug_img)


def detect_lane(camera_stream: cv2.VideoCapture, **kwargs):
    while camera_stream.isOpened():
        _, img = camera_stream.read()
        if img is None:
            raise RuntimeError('No image capture from stream')
        img, img_gray, masked_img, masked_img_gray = process_input(img)
        h, w = img.shape[:2]
        edge_img = get_edge_image(masked_img, masked_img_gray)
        lines = cv2.HoughLinesP(edge_img, 2, np.pi / 180, 100, np.array([]), minLineLength=5, maxLineGap=150)
        left_lane = detect_lane_on_side(lines, w, h, side='left')
        right_lane = detect_lane_on_side(lines, w, h, side='right')

        if len(left_lane) == len(right_lane) == 4:
            intersect_pt = intersection(left_lane, right_lane)
            left_intersect_pt = intersection(left_lane, [0, intersect_pt[1] + 20, w, intersect_pt[1] + 20])
            right_intersect_pt = intersection(right_lane, [0, intersect_pt[1] + 20, w, intersect_pt[1] + 20])
            mask_poly = np.array([
                [left_lane[:2], left_intersect_pt[:2], right_intersect_pt[:2], right_lane[:2]]
            ], dtype=np.int32)

            if kwargs['debug']:
                debug(img, mask_poly)

        cv2.waitKey()

    cv2.destroyAllWindows()
