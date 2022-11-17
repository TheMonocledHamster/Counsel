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
from service_chain.component import Component


class CustomEnv(gym.Env):
    def __init__(
                self, log_dir:str, graph_encoder:str,
                budget:list[int], slo_latency:float, 
                overrun_lim:float, steps_per_epoch:int=8192
                ):

        self.log_dir = log_dir
        self.graph_encoder = graph_encoder

        if 0 in budget:
            raise ValueError('Budget cannot be 0')
        self.budget = budget
        self.slo_latency = slo_latency
        self.overrun_lim = overrun_lim
        self.act_type = 0
        self.act_comp = 0

        self.chain = Chain()
        self._preprocess()

        self.action_space = Discrete(self._num_actions())
        print("act_space size: {}".format(self.action_space.n))
        obs,_ = self.get_obs(0)
        self.observation_space = gym.Space(shape=list(obs.shape))
        print("obv_space size: {}".format(self.observation_space.shape))

        self.action_counter = 0
        self.action_list = []
        self.epoch_counter = 0
        self.steps_per_epoch = steps_per_epoch

        self.epoch_reward = 0
        self.best_epoch = 0


    def _preprocess(self)->None:
        flavors_file = os.path.join(os.path.dirname(__file__),
                                './configs/flavors.json')
        self.flavors = list(dict(json.load(open(flavors_file))).keys())
        conf_file = os.path.join(os.path.dirname(__file__),
                                    './configs/initial_chain.json')
        init_conf = json.load(open(conf_file))
        self.chain.init_components(init_conf, self.budget)


    def _num_actions(self)->list[int]:
        return len(self.flavors)


    def get_obs(self,
                comp:Component=None
                )->tuple[np.ndarray,np.array]:
        
        E_origin = self.chain.get_adj_matrix()
        E_hat = E_origin + np.eye(E_origin.shape[0])

        D = np.diag(np.sum(E_hat, axis=1))
        D_spectral = np.sqrt(np.linalg.inv(D))
        E = np.matmul(np.matmul(D_spectral, E_hat), D_spectral)
        F = self.chain.get_features()
        F[self.act_comp][0] = comp.util
        F[self.act_comp][3] = 1

        ob = np.concatenate((E, F), axis=1)

        mask = np.ones([self.action_space.n])
        if comp is not None:
            instances = comp.get_instances()
            for i in range(self.action_space.n):
                if self.flavors[i] not in instances:
                    mask[i] = 0

        return ob, mask


    def calculate_reward(self)->float:
        u_k = 0
        u_i = 0
        m = 0
        for i,comp in enumerate(self.chain.components.values()):
            if i != self.act_comp:
                u_k += comp.util
            else:
                u_i = comp.util
        for i in range(m):
            if i != self.act_comp:
                u_k += self.chain.get_adj_matrix()[self.act_comp][i]
        return 0.0


    def step(self,action)->None:
        MIN_REW = 1e-8
        obs = None
        mask = None
        reward = None
        done = False
        info = {}

        invalid_flag = False
        self.action_counter += 1

        act_flavor = int(action)

        comp = self.chain.components[self.act_comp]
        invalid_flag = (comp.add_instance(act_flavor) if self.act_type
                         else comp.del_instance(act_flavor))

        # Begin TODO: Comms with Controller here

        latency = 1.34
        self.act_type = 0
        self.act_comp = 0
        arrival_rate = np.full(len(self.chain.components), 100)
        service_rate = np.full(len(self.chain.components), 95)

        # End TODO
        
        for c, lambda_t, mu_t in zip(list(self.chain.components.values()),
                                        arrival_rate,service_rate):
            # Queueing theory Utilization
            # lambda_t = arrival rate at time t
            # mu_t = service rate at time t
            # rho = lambda_t / mu_t * n 
            c.update_util(lambda_t, mu_t)

        # TTL check needed for removal
        if self.act_type == 0:
            obs, mask = self.get_obs(comp)
        else:
            obs, mask = self.get_obs()

        # check for violations
        if latency > self.slo_latency:
            slo_flag = True
        if self.chain.get_budget_overrun() > self.overrun_lim:
            budget_flag = True
        
        if slo_flag and not budget_flag:
            reward = MIN_REW
        elif budget_flag and not slo_flag:
            reward = MIN_REW
        elif slo_flag and budget_flag:
            reward = MIN_REW**2
        elif invalid_flag:
            reward = 100 * MIN_REW
        else:
            # slo_pres = self.slo_latency/(latency + sys.float_info.min)
            reward = max(reward, self.calculate_reward(self.act_comp))

        if self.action_counter % self.steps_per_epoch == 0:
            self.epoch_counter += 1
            done = True

        self.epoch_reward += reward

        return obs, mask, reward, done, info


    def reset(self)->None:
        self.chain.reset()
        self.action_counter = 0
        self._preprocess()
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

    print(env.get_obs()[0], env.get_obs()[1],sep='\n\n')
    # print()
    # print(chain.get_budget_overrun())
    # print()
    # print(chain.get_feasible_actions(
    #     2 * len(chain.components) * len(chain.flavors_list)
    # ))
    # print()
