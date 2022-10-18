import gym

class CustomEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
    
    def preprocess(self):
        pass

    def step(self):
        self
    
    def reset(self):
        pass

    def get_obs(self):
        pass

    def terminate(self):
        pass

    def save_if_best(self):
        pass

    def is_visited(self):
        pass

    def save_trajectory(self):
        pass
