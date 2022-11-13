import json
import math
import os
import sys
from copy import deepcopy
from time import time

import gym
import gym.spaces
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from service_chain.chain import Chain


class CustomEnv(gym.Env):
    def __init__(
                self, chain:Chain, log_dir:str, graph_encoder:str,
                budget:list[int], slo_latency:float, alpha:float,
                max_actions:int, steps_per_epoch:int=2048
                ):

        self.log_dir = log_dir
        self.graph_encoder = graph_encoder
        self.max_actions = max_actions
        self.steps_per_epoch = steps_per_epoch
        self.budget = budget
        self.slo_latency = slo_latency

        self.chain = chain
        self.original_chain = deepcopy(chain)

        self._preprocess()

        obs,_ = self.get_obs()
        self.observation_space = gym.Space(shape=list(obs.shape))
        print("obv_space size: {}".format(self.observation_space.shape))
        self.action_space = gym.spaces.Discrete(self.get_num_actions(),start=1)
        print("act_space size: {}".format(self.action_space.n))


    def _preprocess(self)->None:
        pass


    def get_num_actions(self)->int:
        return self.action_space.n


    def get_obs(self)->np.ndarray:
        E_origin = self.chain.adj_matrix
        E_hat = E_origin + np.eye(E_origin.shape[0])

        D = np.diag(np.sum(E_hat, axis=1))
        D_spectral = np.sqrt(np.linalg.inv(D))
        E = np.matmul(np.matmul(D_spectral, E_hat), D_spectral)
        F = self.chain.feature_matrix

        ob = np.concatenate((E, F), axis=1)
        return ob


    def get_budget_overrun(self):
        pass

    def step(self,action)->None:
        pass
    

    def reset(self)->None:
        pass



    def terminate(self)->None:
        pass


    def save_if_best(self)->None:
        pass


    def save_trajectory(self)->None:
        pass


if __name__ == "__main__":
    chain = Chain(budget=[100, 120])
    env = CustomEnv(chain, log_dir="test", graph_encoder="GCN",
                    budget=[100, 120], slo_latency=0.1, alpha=0.05,
                    max_actions=512, steps_per_epoch=2048)
    print(env.action_space.n)
