import json
import os
import sys
from uuid import uuid4
from typing import List, Tuple
import csv

import gym
from gym.spaces import Discrete
import numpy as np

from .service_chain.chain import Chain
from .service_chain.component import Component
from .synthetic import call_load_server, set_base


class CloudEnv(gym.Env):
    def __init__(
                self, log_dir:str, steps_per_epoch:int,
                budget:List[int], slo_latency:float,
                overrun_lim:float, mode:str='synthetic',
                nconf:int=5, ncomp:int=3
                ):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, uuid4().hex+'.csv')
        with open(self.log_path, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'steps', 'actions', 'episode_reward'])
        if mode not in ['synthetic', 'live']:
            raise ValueError('Invalid mode')
        self.mode = mode
        if 0 in budget:
            raise ValueError('Budget cannot be 0')
        self.budget = budget
        self.slo_latency = slo_latency
        self.overrun_lim = overrun_lim
        self.act_type = 0
        self.act_comp = 0
        self.nconf = nconf
        self.ncomp = ncomp

        self.chain = Chain()
        self.__preprocess()
        set_base(self.components)

        self.action_space = Discrete(self.__num_actions())
        print("act_space size: {}".format(self.action_space.n))
        obs,_ = self.get_obs()
        self.observation_space = gym.Space(shape=list(obs.shape))
        print("obv_space size: {}".format(self.observation_space.shape))

        self.step_counter = 0
        self.prev_steps = 0
        self.action_counter = 0
        self.epoch_counter = 0
        self.steps_per_epoch = steps_per_epoch
        self.episode_counter = 0
        self.episode_reward = 1e-8
        self.BASE_RWD = 1e-4
        self.epoch_done = False
        

    def __preprocess(self)->None:
        flavors_file = os.path.join(os.path.dirname(__file__),
                                f'./configs/flavors{self.nconf}.json')
        self.flavors = list(dict(json.load(open(flavors_file))).items())
        conf_file = os.path.join(os.path.dirname(__file__),
                                f'./configs/initial_chain{self.ncomp}.json')
        init_conf = json.load(open(conf_file))
        self.chain.init_components(init_conf, self.budget, self.nconf)
        self.components:List[Component] = list(self.chain.components.values())
        for c in self.components:
            c.compute_resources()
            c.update_util(1,1)


    def __num_actions(self)->List[int]:
        return len(self.flavors)


    def get_obs(self,
                comp:Component=None,
                arrival_rate:float=0.0
                )->Tuple[np.ndarray,np.array]:
        
        E_origin = self.chain.get_adj_matrix()
        E_hat = E_origin + np.eye(E_origin.shape[0])
        D = np.diag(np.sum(E_hat, axis=1))
        D_spectral = np.sqrt(np.linalg.inv(D))
        E = np.matmul(np.matmul(D_spectral, E_hat), D_spectral)
        F = self.chain.get_features()
        F[:,2] = [comp.util for comp in self.components]
        F[:,3] = arrival_rate
        F[self.act_comp][4] = 1

        ob = np.concatenate((E, F), axis=1)

        mask = np.ones([self.action_space.n])
        if comp is not None and self.act_type == 0:
            instances = comp.get_instances()
            for i in range(1,self.action_space.n):
                if self.flavors[i][0] not in instances:
                    mask[i] = 0

        return ob, mask


    def calculate_reward(self, action:int)->float:
        comp:Component = self.components[self.act_comp]
        upsilon_i = min(comp.util, 1)
        upsilon_k = min(sum([ocomp.util for ocomp in self.components]),
                        len(self.components)) - upsilon_i

        alpha_cpu = (self.budget[0] - comp.cpu)
        alpha_mem = (self.budget[1] - comp.mem)
        cpu_, mem_ = self.flavors[action][1][:]

        # Utility functions
        sgn = lambda x: (x>0)-(x<0)
        dim_rwd = lambda a,x: sgn(a-x)*abs(1-x/a)

        # Reward
        rwd_cpu = dim_rwd(alpha_cpu + 1e-8, cpu_)
        rwd_mem = dim_rwd(alpha_mem + 1e-8, mem_)
        reward = upsilon_k + upsilon_i*(rwd_cpu + rwd_mem)/2
        reward = max(0.01, reward)

        # Max Possible Reward = comp_count
        return reward


    def step(self,action)->None:
        invalid_flag = False
        info = {}

        self.step_counter += 1
        action = int(action)
        act_flavor = self.flavors[action][0]

        comp = self.components[self.act_comp]

        if action != 0 and self.act_comp != -1:
            _ = (comp.add_instance(act_flavor) if self.act_type
                            else comp.del_instance(act_flavor))
        
        self.action_counter += min(action, 1)

        # Sync call
        if self.mode == 'synthetic':
            metrics = call_load_server([comp.cpu for comp in self.components],
                                       [comp.mem for comp in self.components])
        else:
            metrics = None #TODO: Call Orchestrator

        arrival_rate = metrics[0]
        cutils = metrics[1]
        mutils = metrics[2]
        latency = metrics[3]
        self.act_type = metrics[4]
        self.act_comp = metrics[5]
        done = metrics[6]

        for comp,cutil,mutil in zip(self.components,cutils,mutils):
            comp.update_util(cutil, mutil)

        comp = self.components[self.act_comp]
        obs, mask = self.get_obs(comp, arrival_rate)

        reward = self.BASE_RWD

        # check for violations
        if (lat_viol:=(latency/self.slo_latency)) > 1:
            reward **= lat_viol
        overrun = self.chain.get_budget_overrun()
        if (b_viol:=(overrun/(self.overrun_lim+sys.float_info.min))) > 1:
            reward **= (b_viol+1)/2
        if invalid_flag:
            reward *= 0.5
        if reward == self.BASE_RWD:
            reward = self.calculate_reward(action) # Guaranteed reward >= 0.01
        
        reward = max(1e-40, reward)
        
        self.episode_reward += reward

        if self.step_counter % self.steps_per_epoch == 0:
            self.epoch_counter += 1
            self.epoch_done = True
        
        if self.epoch_done and self.step_counter > self.ncomp * 10:
            done = True

        if done:
            self.episode_counter += 1
            with open(self.log_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_counter,
                                 self.step_counter - self.prev_steps,
                                 self.action_counter,
                                 self.episode_reward])
            self.prev_steps = self.step_counter
            self.episode_reward = 1e-8
            self.action_counter = 0

        return obs, mask, reward, done, info


    def reset(self)->None:
        self.chain.reset()
        self.__preprocess()
        self.epoch_done = False
        return self.get_obs()


    def terminate(self)->None:
        # TODO ?
        pass
