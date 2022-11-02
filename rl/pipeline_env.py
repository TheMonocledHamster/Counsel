import gym

class Pipeline(gym.Env):
    def __init__(self) -> None:
        pass
    
    def preprocess(self)->None:
        pass

    def step(self)->None:
        pass
    
    def reset(self)->None:
        pass

    def get_obs(self)->None:
        pass

    def terminate(self)->None:
        pass

    def save_if_best(self)->None:
        pass

    def is_visited(self)->None:
        pass

    def save_trajectory(self)->None:
        pass
