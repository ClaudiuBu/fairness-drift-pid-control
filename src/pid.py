class PIDController:
    def __init__(self, kp, ki, kd, target=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target

        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, current_value):
        error = self.target - current_value
        self.integral += error
        self.integral = max(-50, min(self.integral, 50))
        derivative = error - self.prev_error
        self.prev_error = error

        control = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
        return control