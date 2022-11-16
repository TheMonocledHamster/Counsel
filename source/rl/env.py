import json
import math
import os
import sys
from copy import deepcopy
from time import time

import gym
from gym.spaces import Discrete, Box
import networkx as nx
import numpy as np

from service_chain.chain import Chain


class CustomEnv(gym.Env):
    def __init__(
                self, chain:Chain, log_dir:str, graph_encoder:str,
                budget:list[int], slo_latency:float, 
                overrun_lim:float, steps_per_epoch:int=2048
                ):

        self.log_dir = log_dir
        self.graph_encoder = graph_encoder

        self.budget = budget
        self.slo_latency = slo_latency
        self.overrun_lim = overrun_lim
        self.latency = 0
        self.flavor_num = 0

        self.chain = chain
        self._preprocess()

        self.action_space = Discrete(self._num_actions(), start=1)
        print("act_space size: {}".format(self.action_space.n))
        obs,_ = self.get_obs()
        self.observation_space = gym.Space(shape=list(obs.shape))
        print("obv_space size: {}".format(self.observation_space.shape))

        self.act_type = -1
        self.act_comp = -1

        self.action_counter = 0
        self.action_list = []
        self.epoch_counter = 0
        self.steps_per_epoch = steps_per_epoch

        self.epoch_reward = 0
        self.best_epoch = 0


    def _preprocess(self)->None:
        flavors_file = os.path.join(os.path.dirname(__file__), 
                                '../configs/flavors.json')
        flavors = dict(json.load(open(flavors_file))).keys()
        self.flavor_num = len(flavors)

        conf_file = os.path.join(os.path.dirname(__file__),
                                    '../configs/initial_chain.json')
        init_conf = json.load(open(conf_file))
        self.chain.init_components(init_conf, flavors, self.budget)
        self.comp_num = len(self.chain.components)


    def _num_actions(self)->list[int]:
        return self.flavor_num


    def get_obs(self, 
                act_type:int=-1, 
                act_comp:int=-1
                )->tuple[
                    np.ndarray,
                    np.array
                ]:
        E_origin = self.chain.get_adj_matrix()
        E_hat = E_origin + np.eye(E_origin.shape[0])

        D = np.diag(np.sum(E_hat, axis=1))
        D_spectral = np.sqrt(np.linalg.inv(D))
        E = np.matmul(np.matmul(D_spectral, E_hat), D_spectral)
        F = self.chain.get_features()

        ob = np.concatenate((E, F), axis=1)

        mask = np.ones([self.action_space.n])
        if act_type == 0:
            mask = np.asarray(self.chain.get_feasible_actions(mask))

        return ob, mask


    def calculate_reward(self)->float:
        reward = 1e-5
        return reward


    def step(self,action)->None:
        MIN_REW = 1e-8
        obs = None
        mask = None
        reward = None
        done = False
        info = {}

        self.action_counter += 1

        action = int(action)
        # TODO Redo this
        act_type = 0 if action <= self.action_space.n/2 else 1
        act_comp = (int(action/self.flavor_num)+1) if act_type == 0 \
                   else int((action-self.action_space.n/2 -1)/self.flavor_num)
        act_flavor = int(action % self.flavor_num)
        self.action_list.append(act_type, act_comp, act_flavor)

        if act_type == 0:
            self.chain.components[act_comp].remove_instances(act_flavor)
        else:
            self.chain.components[act_comp].add_instances(act_flavor)

        # Wait Here

        latency = 1.34
        act_type, act_comp = 0,0 # TODO Receive from the controller
        obs, mask = self.get_obs(act_type,act_comp)

        # check budget violation and slo violation
        if latency > self.slo_latency:
            reward = MIN_REW
        elif (overrun:=self.chain.get_budget_overrun()) > self.overrun_lim:
            reward = MIN_REW
        else:
            slo_pres = self.slo_latency/(latency + sys.float_info.min)
            reward = max(reward, self.calculate_reward(overrun,slo_pres))

        if self.action_counter % self.steps_per_epoch == 0:
            self.epoch_counter += 1
            done = True

        self.epoch_reward += reward

        return obs, mask, reward, done, info


    def reset(self)->None:
        self.chain.reset()
        self.action_counter = 0
        self._preprocess()
        sys.stdout.flush()
        return self.get_obs()


    def terminate(self)->None:
        pass


    def save_if_best(self)->None:
        pass


    def save_trajectory(self)->None:
        pass


if __name__ == "__main__":
    chain = Chain()
    env = CustomEnv(chain, log_dir="test", graph_encoder="GCN",
                    budget=[125, 550], slo_latency=0.1,
                    overrun_lim=0.05, steps_per_epoch=2048)

    print(env.get_obs()[0], env.get_obs()[1],sep='\n')
    print(chain)
    # print()
    # print(chain.get_budget_overrun())
    # print()
    # print(chain.get_feasible_actions(
    #     2 * len(chain.components) * len(chain.flavors_list)
    # ))
    # print()
