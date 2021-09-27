import os.path as path
import lane_detection

if __name__ == '__main__':
    current_dir = path.dirname(__file__)
    input_dir = path.join(current_dir, 'input', 'lane')
    test_file = path.join(input_dir, 'Lane Detection Test Video 01.mp4')
    lane_detection.detect_lane(test_file)
