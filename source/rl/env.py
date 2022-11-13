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
from source.service_chain.chain import Chain


class CustomEnv(gym.Env):
    def __init__(
                self, chain:Chain, log_dir:str, graph_encoder:str,
                budget:list[int], slo_latency:float, 
                alpha_lim:float, steps_per_epoch:int=2048
                ):

        self.log_dir = log_dir
        self.graph_encoder = graph_encoder

        self.budget = budget
        self.slo_latency = slo_latency
        self.latency = 0
        self.alpha_lim = alpha_lim

        self.chain = chain
        self._preprocess()
        self.max_node = len(self.chain.components)


        obs = self.chain_repr()
        self.observation_space = gym.Space(shape=list(obs.shape))
        print("obv_space size: {}".format(self.observation_space.shape))
        self.action_space = gym.spaces.Discrete(self._num_actions())
        print("act_space size: {}".format(self.action_space.n))

        self.action_counter = 0
        self.epoch_counter = 0
        self.steps_per_epoch = steps_per_epoch

        self.epoch_reward = 0
        self.best_epoch = 0


    def _preprocess(self)->None:
        file_path = os.path.join(os.path.dirname(__file__),
                                    '../configs/initial_chain.json')
        init_conf = json.load(open(file_path))

        self.chain.init_components(init_conf)


    def _num_actions(self)->int:
        return None

    def get_latency(self)->float:
        self.latency = 0
        return self.latency


    def chain_repr(self)->np.ndarray:
        E_origin = self.chain.adj_matrix
        E_hat = E_origin + np.eye(E_origin.shape[0])

        D = np.diag(np.sum(E_hat, axis=1))
        D_spectral = np.sqrt(np.linalg.inv(D))
        E = np.matmul(np.matmul(D_spectral, E_hat), D_spectral)
        F = self.chain.feature_matrix

        ob = np.concatenate((E, F), axis=1)
        return ob


    def step(self,action)->None:
        obs = None
        reward = 1e-5
        done = False

        bud_viol_flag,  = False
        slo_viol_flag,  = False

        self.action_counter += 1
        alpha = 0

        # check budget violation and slo violation
        if (alpha!=self.chain.get_budget_overrun()) > self.alpha_lim:
            bud_viol_flag = True
        if self.get_latency() > self.slo_latency:
            slo_viol_flag = True
        
        reward += (1-alpha)*() - alpha*()

        # if budget violation or slo violation
        if bud_viol_flag or slo_viol_flag:
            reward = 1e-5

        if self.action_counter % self.steps_per_epoch == 0:
            self.epoch_counter += 1
            done = True

        self.epoch_reward += reward
        return obs, reward, done
    

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
