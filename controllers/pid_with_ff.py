from . import BaseController
import numpy as np

class LowPassFilter:
    def __init__(self, coefficient):
        self.coefficient = coefficient
        self.prev_value = 0

    def filter(self, current_value):
        filtered_value = (self.coefficient * current_value) + ((1 - self.coefficient) * self.prev_value)
        self.prev_value = filtered_value
        return filtered_value

class Controller(BaseController):

    def __init__(self):
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0
        self.aw_integral_limit = 8.

        self.ff_weight = 0.1
        self.low_pass_filter = LowPassFilter(0.9)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        self.error_integral = max(
            min(self.error_integral, self.aw_integral_limit), -self.aw_integral_limit)

        error_diff = error - self.prev_error
        smoothed_diff = self.low_pass_filter.filter(error_diff)
        self.prev_error = error

        feedforward = self.compute_feedforward2(state, future_plan)

        output = self.p * error + self.i * self.error_integral + self.d * smoothed_diff + feedforward

        return output
    
    def compute_feedforward2(self, state, future_plan):
        feedforward = 0
        if future_plan and future_plan.lataccel:
            if len(future_plan.lataccel) > 0:
                predicted_lataccel = future_plan.lataccel[0]                
                delta_lataccel = predicted_lataccel - state.roll_lataccel
                
                velocity_factor = future_plan.v_ego[0] / state.v_ego if state.v_ego != 0 else 1
                accel_factor = future_plan.a_ego[0] - state.a_ego
                
                feedforward = self.ff_weight * (predicted_lataccel \
                                                + delta_lataccel* velocity_factor\
                                                    #   + accel_factor\
                                                        )        
        return feedforward