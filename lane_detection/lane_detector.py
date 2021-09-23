# receive input
import cv2
import cv2.cv2
import numpy
import numpy as np
from numpy import ndarray


def detect_lane(file_path: str):
    print(file_path)
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        _, img = cap.read()
        # img = cv2.imread(file_path)
        img = cv2.resize(img, (750, 562))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)

        #   b, r, g = cv2.split(img)
        # (thresh, img_bi) = cv2.threshold(img_gray, 100, 200, cv2.THRESH_BINARY)

        # edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        width = canny.shape[1]  # image dim: 562x750
        height = canny.shape[0]
        # TODO: resize image to same dim, identify left and right lane

        polygon = np.array(
            [[(0, height), (0, 4 * height / 5), (3*width / 4, 3*height / 4), (width, 4 * height / 5), (width, height)]],
            np.int)

        mask = np.zeros_like(canny)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(canny, mask)
        masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        # masked_image = cv2.erode(masked_image, np.ones((3, 3), np.uint8), 1)

        lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)
        # print(len(lines))
        # for i in range(0, len(lines) - 1):
        # lines = np.delete(lines, 0, 0)
        # print(lines)
        left_fit = []
        right_fit = []
        mid = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line.reshape(4)
            # if abs(x1 - x2) < 10:
            #     newlines = np.delete(lines, i, 0)
            #     mid.append(i)
            #     print(mid)
            parameter = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameter[0]
            intercept = parameter[1]
            if abs(x1 - x2) < 15:
                mid.append((slope, intercept))
            elif slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        mid_average = np.average(mid, axis=0)

        # print([left_line, right_line])
        def lane(side):
            y1 = height
            y2 = int(y1 * 0.8)
            x1 = int((y1 - side[1]) / side[0])
            x2 = int((y2 - side[1]) / side[0])
            return np.array([x1, y1, x2, y2])

        right_line = lane(right_fit_average)
        left_line = lane(left_fit_average)
        # mid_line = lane(mid_average)

        img_line = np.zeros_like(img)
        # for line in lines:
        for x1, y1, x2, y2 in np.array([left_line, right_line]):
            cv2.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 5)
        img_line1 = np.zeros_like(img)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_line1, (x1, y1), (x2, y2), (0, 0, 255), 1)
        img_final = cv2.addWeighted(img, 0.8, img_line, 1, 1)

        cv2.imshow('canny', canny)
        cv2.imshow('masked', masked_image)
        cv2.imshow('lines', img_line1)
        cv2.imshow('line', img_line)
        cv2.imshow('window', img_final)
        cv2.waitKey(1)

    cap.release()

    cv2.destroyAllWindows()
