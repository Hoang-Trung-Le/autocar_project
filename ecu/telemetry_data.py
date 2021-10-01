class TelemetryData:
    steering_angle: float
    speed: float

    def __init__(self, steering_angle: float, speed: float):
        self.steering_angle = steering_angle
        self.speed = speed

    @classmethod
    def sample_data(cls):
        """
        Create a sample telemetry data for testing
        :return: TelemetryData instance
        """
        return TelemetryData(steering_angle=0, speed=60)
