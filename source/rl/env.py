from time import time
import gym
from copy import deepcopy
import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

from service_chain.chain import Chain


class CustomEnv(gym.Env):
    def __init__(self, chain:Chain, log_dir:str, graph_encoder:str, budget,
                flavor_count, max_actions=512, steps_per_epoch=2048) -> None:

        self.log_dir = log_dir
        self.max_actions = max_actions
        self.steps_per_epoch = steps_per_epoch
        self.budget = budget

        self.w1 = 2
        self.w2 = 2
        self.normal_param = 1e-7
        self.graph_encoder = graph_encoder
        self.max_node = len(chain.components)

        self.chain = chain
        self.original_chain = deepcopy(chain)

        obs,_ = self.get_obs()
        self.observation_space = gym.Space(shape=list(obs.shape))
        print("obv_space size: {}".format(self.observation_space.shape))
        self.action_space = gym.spaces.Discrete(self.compute_action_space())
        print("act_space size: {}".format(self.action_space.n))

        self.max_reward = None
        self.cum_reward = 0
        self.complete_count = 0
        self.action_count = 0
        self.act_list = []
        self.epoch_idx = 0
        self.cost = 0

        self.optm_target = None
        self.optm_cost = 0
        self.optm_chain = self.chain
        self.optm_ob = None
        self.optm_act_list = []
        self.optm_epoch_idx = 0

        self.start_time = int(time())

        actions_path = "results/{}/actions.txt".format(self.log_dir)
        self.action_fptr = open(actions_path, "w")
        self.chain_path = "results/{}/optm_chain".format(self.log_dir)
        if not os.path.exists(self.chain_path):
            os.makedirs(self.chain_path)
        
        self.cum_act_count = 0 # Record Epoch Number
        self.traj_set = set()
        self.main_traj_stats = []
        traj_path = "results/{}/traj.txt".format(self.log_dir)
        self.traj_fptr = open(traj_path, "w")
        

    def compute_action_space(self)->None:
        pass


    def get_feasible_actions(self)->None:
        pass

    
    def step(self,action)->None:
        obs, reward, done, info = None, None, False, None

        violation_flag, visited_flag = False, False

        act_int, act_type = int(action), 0

        obs, mask = self.get_obs()
        if sum(mask) == 0:
            violation_flag = True
        
        if self.action_count > self.max_actions or violation_flag:
            done = True
        else:
            done = False
        
        if done or self.cum_act_count%self.steps_per_epoch == 0:
            visited_flag = self.is_visited()
        
        self.cum_reward += reward
    

    def reset(self)->None:
        self.action_count = 0
        self.cum_reward = 0
        self.cost = 0

        self.chain.reset()
        self.epoch_idx += 1
        self.act_list = []

        return self.get_obs()


    def get_obs(self)->None:
        E_origin = self.chain.adj_matrix
        E_hat = E_origin + np.eye(E_origin.shape[0])

        D = np.diag(np.sum(E_hat, axis=1))
        D_spectral = np.sqrt(np.linalg.inv(D))
        E = np.matmul(np.matmul(D_spectral, E_hat), D_spectral)
        F = self.chain.feature_matrix
        
        ob = np.concatenate((E, F), axis=1)
        mask = np.asarray(self.get_feasible_actions())
        return ob, mask


    def terminate(self)->None:
        self.action_fptr.write("Epoch Count: {}, node_num:{}\n"\
            .format(self.epoch_idx, self.max_node))
        self.action_fptr.write("Time: {}s\n".format(int(time())
                                            - self.start_time))


    def save_if_best(self)->None:
        self.complete_count += 1
        # Logic

    def is_visited(self)->None:
        link_cand_list = [int(action) for (action, cost) in self.act_list]
        link_cand_tuple = tuple(sorted(link_cand_list))
        visited_flag = (tuple(link_cand_tuple) in self.traj_set)


    def save_trajectory(self)->None:
        main_epoch_idx = int((self.cum_act_count-1)/self.steps_per_epoch)
