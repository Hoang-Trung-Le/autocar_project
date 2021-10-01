import lane_detection
from ecu.telemetry import read_front_camera_stream

if __name__ == '__main__':
    front_camera_stream = read_front_camera_stream()
    try:
        lane_detection.detect_lane(front_camera_stream, debug=True)
    finally:
        front_camera_stream.release()

