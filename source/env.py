import json
import os
import sys
from typing import List, Tuple

import gym
from gym.spaces import Discrete
import numpy as np

from service_chain.chain import Chain
from service_chain.component import Component


class CustomEnv(gym.Env):
    def __init__(
                self, log_dir:str, graph_encoder:str,
                budget:List[int], slo_latency:float, 
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
        obs,_ = self.get_obs()
        self.observation_space = gym.Space(shape=list(obs.shape))
        print("obv_space size: {}".format(self.observation_space.shape))

        self.action_counter = 0
        self.epoch_counter = 0
        self.steps_per_epoch = steps_per_epoch

        self.epoch_reward = 0
        self.BASE_RWD = 1e-4
        self.best_epoch = 0
    

    def _preprocess(self)->None:
        flavors_file = os.path.join(os.path.dirname(__file__),
                                './configs/flavors.json')
        self.flavors = list(dict(json.load(open(flavors_file))).items())
        conf_file = os.path.join(os.path.dirname(__file__),
                                    './configs/initial_chain.json')
        init_conf = json.load(open(conf_file))
        self.chain.init_components(init_conf, self.budget)
        self.components = list(self.chain.components.values())
        for c in self.components:
            c.update_util(1,1)


    def _num_actions(self)->List[int]:
        return len(self.flavors)


    def get_obs(self,
                comp:Component=None
                )->Tuple[np.ndarray,np.array]:
        
        E_origin = self.chain.get_adj_matrix()
        E_hat = E_origin + np.eye(E_origin.shape[0])
        D = np.diag(np.sum(E_hat, axis=1))
        D_spectral = np.sqrt(np.linalg.inv(D))
        E = np.matmul(np.matmul(D_spectral, E_hat), D_spectral)
        F = self.chain.get_features()
        F[:,0] = [comp.util for comp in self.components]
        F[self.act_comp][3] = 1

        ob = np.concatenate((E, F), axis=1)

        mask = np.ones([self.action_space.n])
        if comp is not None:
            instances = comp.get_instances()
            for i in range(self.action_space.n):
                if self.flavors[i][0] not in instances:
                    mask[i] = 0

        return ob, mask


    def calculate_reward(self)->float:
        comp = self.components[self.act_comp]
        upsilon_i = comp.util
        upsilon_k = sum([ocomp.util for ocomp in self.components if ocomp != comp])
        
        alpha_cpu = (self.budget[0] - comp.cpu)
        alpha_mem = (self.budget[1] - comp.mem)
        cpu_, mem_ = self.flavors[self.act_type][1][:]

        # Utility functions
        sgn = lambda x: (x>0)-(x<0)
        dim_rwd = lambda a,x: sgn(a-x)*abs(1-x/a)

        # Reward
        rwd_cpu = dim_rwd(alpha_cpu, cpu_)
        rwd_mem = dim_rwd(alpha_mem, mem_)
        reward = upsilon_k + upsilon_i*(rwd_cpu + rwd_mem)/2
        reward = max(0.01, reward)

        return reward


    def step(self,action)->None:
        obs = None
        mask = None
        reward = None
        done = False
        info = {}

        self.action_counter += 1

        act_flavor = int(action)

        comp = self.components[self.act_comp]
        invalid_flag = (comp.add_instance(act_flavor) if self.act_type
                        else comp.del_instance(act_flavor))

        # Begin TODO: Comms with Controller here

        latency = 1.34
        self.act_type = 0
        self.act_comp = 0
        arrival_rate = np.full(len(self.components), 100)
        service_rate = np.full(len(self.components), 95)

        # End TODO
        
        for c, lambda_t, mu_t in \
            zip((self.components),arrival_rate,service_rate):
            """
                Queueing Theory Utilization 
                G/G/m
                lambda_t = arrival rate at time t
                mu_t = service rate at time t
                rho = lambda_t / mu_t * m
            """
            c.update_util(lambda_t, mu_t)

        # TTL check needed for removal
        self.get_obs(comp)

        reward = self.BASE_RWD
        
        # check for violations
        if (lat_viol:=latency/self.slo_latency) > 1:
            reward **= lat_viol
        overrun = self.chain.get_budget_overrun()
        if (b_viol:=overrun/(self.overrun_lim+sys.float_info.min)) > 1:
            reward **= (b_viol+1)/2
        if invalid_flag:
            reward *= 0.5
        if reward == self.BASE_RWD:
            reward = self.calculate_reward() # Guaraneteed reward >= 0.01

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



if __name__ == "__main__":
    env = CustomEnv(log_dir="test", graph_encoder="GCN",
                    budget=[125, 550], slo_latency=0.1,
                    overrun_lim=0.05, steps_per_epoch=2048)
    comp = env.components[0]
    print(env.get_obs(comp)[0], env.get_obs(comp)[1],sep='\n\n')
    # print()
    # print(chain.get_budget_overrun())
    # print()
    # print(chain.get_feasible_actions(
    #     2 * len(chain.components) * len(chain.flavors_list)
    # ))
    # print()
