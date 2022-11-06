import gym
from copy import deepcopy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

from pipeline.pipeline import Pipeline


class Pipeline(gym.Env):
    def __init__(self, pipeline:Pipeline, log_dir, graph_encoder, \
        max_actions=512, steps_per_epoch=2048) -> None:
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
