import gym
from copy import deepcopy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

from pipeline.pipeline import Pipeline


class Pipeline(gym.Env):
    def __init__(self, pipeline:Pipeline, graph_encoder, \
        max_actions=512, steps_per_epoch=2048) -> None:
        self.max_actions = max_actions
        self.steps_per_epoch = steps_per_epoch
        
        self.max_reward = None
        self.opt_target = None
        self.action_count = 0

        self.graph_encoder = graph_encoder

        self.pipeline = pipeline

    
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
