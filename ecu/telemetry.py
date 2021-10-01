import os.path as path

import cv2

from ecu.telemetry_data import TelemetryData


def read_telemetry():
    """
    Simulate car telemetry
    """
    return TelemetryData.sample_data()


def read_front_camera_stream():
    """
    Simulate reading front camera stream
    :return: Video capture
    """
    current_dir = path.dirname(__file__)
    input_dir = path.join(path.dirname(current_dir), 'input', 'lane')
    test_file = path.join(input_dir, 'Lane Detection Test Video 01.mp4')
    return cv2.VideoCapture(test_file)
