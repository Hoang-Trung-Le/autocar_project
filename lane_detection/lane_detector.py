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
        # canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        #   b, r, g = cv2.split(img)
        # (thresh, img_bi) = cv2.threshold(img_gray, 100, 200, cv2.THRESH_BINARY)

        # edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        width = canny.shape[1]  # image dim: 562x750
        height = canny.shape[0]
        # TODO: resize image to same dim, identify left and right lane

        # polygon = np.array(
        #     [[(0, height), (0, 4 * height / 5), (3 * width / 4, 3 * height / 4), (width, 4 * height / 5),
        #       (width, height)]], np.int)

        roi = np.float32(
            [[0, height], [width / 2 - 100, 1 * height / 2], [width / 2 + 100, 1 * height / 2], [width - 0, height]])
        roi_warp = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
        M = cv2.getPerspectiveTransform(roi, roi_warp)
        warp = cv2.warpPerspective(canny, M, (width, height))
        cv2.imshow('warp', warp)
        histogram = np.sum(warp[int(warp.shape[0] / 2):, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        windows = 10
        window_height = np.int(warp.shape[0] / windows)
        nonzero = warp.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []
        for window in range(windows):
            win_y_low = warp.shape[0] - (window + 1) * window_height
            win_y_high = warp.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_ind = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (
                    nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_ind = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (
                    nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_ind)
            right_lane_inds.append(good_right_ind)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_ind) > minpix:
                leftx_current = np.int(np.mean(nonzero_x[good_left_ind]))
            if len(good_right_ind) > minpix:
                rightx_current = np.int(np.mean(nonzero_x[good_right_ind]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds]
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds]

        left_lane = np.polyfit(lefty, leftx, 2)
        right_lane = np.polyfit(righty, rightx, 2)

        # mask = np.zeros_like(canny)
        # cv2.fillPoly(mask, polygon, 255)
        # masked_image = cv2.bitwise_and(canny, mask)
        # masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        # masked_image = cv2.erode(masked_image, np.ones((3, 3), np.uint8), 1)

        # lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)

        # left_fit = []
        # right_fit = []
        # mid = []
        # for i, line in enumerate(lines):
        #     x1, y1, x2, y2 = line.reshape(4)
        #     # if abs(x1 - x2) < 10:
        #     #     newlines = np.delete(lines, i, 0)
        #     #     mid.append(i)
        #     #     print(mid)
        #     parameter = np.polyfit((x1, x2), (y1, y2), 1)
        #     slope = parameter[0]
        #     intercept = parameter[1]
        #     if abs(x1 - x2) < 15:
        #         mid.append((slope, intercept))
        #     elif slope < 0:
        #         left_fit.append((slope, intercept))
        #     else:
        #         right_fit.append((slope, intercept))
        # left_fit_average = np.average(left_fit, axis=0)
        # right_fit_average = np.average(right_fit, axis=0)
        # mid_average = np.average(mid, axis=0)
        #
        # # print([left_line, right_line])
        # def lane(side):
        #     y1 = height
        #     y2 = int(y1 * 0.8)
        #     x1 = int((y1 - side[1]) / side[0])
        #     x2 = int((y2 - side[1]) / side[0])
        #     return np.array([x1, y1, x2, y2])
        #
        # right_line = lane(right_fit_average)
        # left_line = lane(left_fit_average)
        # mid_line = lane(mid_average)

        # img_line = np.zeros_like(img)
        # # for line in lines:
        # for x1, y1, x2, y2 in np.array([left_lane, right_lane]):
        #     cv2.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 5)
        # img_line1 = np.zeros_like(img)
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(img_line1, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # img_final = cv2.addWeighted(img, 0.8, img_line, 1, 1)

        ploty = np.linspace(0, warp.shape[0] - 1, warp.shape[0])
        left_fitx = left_lane[0] * ploty ** 2 + left_lane[1] * ploty + left_lane[2]
        right_fitx = right_lane[0] * ploty ** 2 + right_lane[1] * ploty + right_lane[2]

        bottom_left = left_lane[0] * height ** 2 + left_lane[1] * height + left_lane[2]
        bottom_right = right_lane[0] * height ** 2 + right_lane[1] * height + right_lane[2]
        bottom_center = (bottom_right + bottom_left) / 2
        center_lane_dev1 = (left_lane[0] + right_lane[0]) * height + (left_lane[1] + right_lane[1])
        center_lane_dev2 = left_lane[0] + right_lane[0]
        curv = ((1 + center_lane_dev1 ** 2) ** 1.5) / abs(center_lane_dev2)
        print(center_lane_dev1)
        warp_zero = np.zeros_like(warp).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        inv_M = cv2.getPerspectiveTransform(roi_warp, roi)
        new_warp = cv2.warpPerspective(color_warp, inv_M, (width, height))

        cv2.putText(img, 'Curvature: ' + str(curv) + ' (pix)', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(img, 'Car is ' + str(width / 2 - bottom_center) + ' (pix) from center lane', (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Combine the result with the original image
        img_final = cv2.addWeighted(img, 1, new_warp, 0.3, 0)

        cv2.imshow('canny', canny)
        # cv2.imshow('masked', masked_image)
        # cv2.imshow('lines', img_line1)
        # cv2.imshow('line', img_line)
        cv2.imshow('window', img_final)
        cv2.waitKey(1)

    cap.release()

    cv2.destroyAllWindows()
