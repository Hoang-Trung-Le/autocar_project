import cv2

import lane_detection
import os.path as path


def read_video_stream():
    current_dir = path.abspath(path.dirname(__file__))
    return cv2.VideoCapture(path.join(current_dir, 'input', 'lane', 'yt5s.com-Lane detect test data.mp4'))


if __name__ == '__main__':
    stream = read_video_stream()
    lane_detection.detect_lane(stream, debug=True)
