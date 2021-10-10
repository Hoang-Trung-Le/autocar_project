import cv2
import cv2.cv2
import numpy as np


def preprocess_image(img):
    img = cv2.resize(img, (750, 562))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return img, img_gray, canny


def interest_area_warp(canny, width, height):
    roi = np.float32(
        [[0, height], [width / 2 - 100, 1 * height / 2], [width / 2 + 100, 1 * height / 2], [width - 0, height]])
    roi_warp = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
    M = cv2.getPerspectiveTransform(roi, roi_warp)
    inv_M = cv2.getPerspectiveTransform(roi_warp, roi)
    warp = cv2.warpPerspective(canny, M, (width, height))
    return warp, M, inv_M


def calculate_curvature(left_lane, right_lane, height):
    center_lane_dev1 = (left_lane[0] + right_lane[0]) * height + (left_lane[1] + right_lane[1])
    center_lane_dev2 = left_lane[0] + right_lane[0]
    curvature = ((1 + center_lane_dev1 ** 2) ** 1.5) / abs(center_lane_dev2)
    return curvature


def offset_to_lane_center(left_lane, right_lane, width, height):
    bottom_left = left_lane[0] * height ** 2 + left_lane[1] * height + left_lane[2]
    bottom_right = right_lane[0] * height ** 2 + right_lane[1] * height + right_lane[2]
    bottom_center = (bottom_right + bottom_left) / 2
    return width / 2 - bottom_center


def find_lanes(warp):
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

    return left_lane, right_lane


def debug(img, warp, inv_M, left_lane, right_lane, width, height, curvature, center_offset):
    warp_zero = np.zeros_like(warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warp.shape[0] - 1, warp.shape[0])
    left_fitx = left_lane[0] * ploty ** 2 + left_lane[1] * ploty + left_lane[2]
    right_fitx = right_lane[0] * ploty ** 2 + right_lane[1] * ploty + right_lane[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, inv_M, (width, height))

    cv2.putText(img, 'Curvature: ' + str(curvature) + ' (pix)', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.putText(img, 'Car is ' + str(center_offset) + ' (pix) from center lane', (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Combine the result with the original image
    img_final = cv2.addWeighted(img, 1, new_warp, 0.3, 0)

    cv2.imshow('window', img_final)
    cv2.waitKey(1)


def detect_lane(stream: cv2.VideoCapture, **kwargs):
    if stream is None:
        return

    try:
        while stream.isOpened():
            _, img = stream.read()
            img, img_gray, canny = preprocess_image(img)
            height, width = canny.shape[:2]

            # TODO: resize image to same dim, identify left and right lane
            warp, _, inv_M = interest_area_warp(canny, width, height)
            left_lane, right_lane = find_lanes(warp)

            center_offset = offset_to_lane_center(left_lane, right_lane, width, height)
            curvature = calculate_curvature(left_lane, right_lane, height)

            if 'debug' in kwargs:
                debug(img, warp, inv_M, left_lane, right_lane, width, height, curvature, center_offset)

        stream.release()
    except Exception as e:
        print('An error has been occurred during frame processing', e)
        cv2.destroyAllWindows()
