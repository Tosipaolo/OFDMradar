class Target:
    def __init__(self, target_rcs, target_distance, target_speed):
        self.target_rcs = target_rcs
        self.target_distance = target_distance
        self.target_speed = target_speed
        self.lightspeed = 3e8
        self.lambda_wv = 10.714e-3  # lamda at 28 GHz
        self.target_delay = 2 * self.target_distance / self.lightspeed
        self.target_doppler = 2 * self.target_speed / self.lambda_wv

# Accessing the fields of the instance
# print("Target RCS:", target_instance.target_rcs)
# print("Target Distance:", target_instance.target_distance)
