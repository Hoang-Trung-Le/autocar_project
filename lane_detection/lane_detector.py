# receive input
import cv2
import cv2.cv2
import numpy as np
from numpy import ndarray


def detect_lane(file_path: str):
    print(file_path)
    img = cv2.imread(file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    #   b, r, g = cv2.split(img)
    (thresh, img_bi) = cv2.threshold(img_gray, 100, 200, cv2.THRESH_BINARY)

    # edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    width = canny.shape[1]  # image dim: 562x750
    height = canny.shape[0]
    polygon = np.array(
        [[(width/2, height), (width / 2, height / 2), (width, height - 100), (width, height)]], np.int)
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    masked_image = cv2.erode(masked_image, np.ones((5, 5), np.uint8), 1)

    lines = cv2.HoughLinesP(masked_image, 6, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=1)
    print(lines)
    img_line = np.zeros_like(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 3)

    img_final = cv2.addWeighted(img, 0.8, img_line, 1, 1)

    cv2.imshow('img', masked_image)
    cv2.imshow('img1', img_line)
    cv2.imshow('window', img_final)
    cv2.waitKey()
    cv2.destroyAllWindows()
