import time
import torch
from typing import Tuple, List

from .env import CustomEnv
from .core import GCNActorCritic
from .ppo import ppo
from .service_chain.chain import Chain
from .utils.mpi_tools import mpi_fork

# self, chain:Chain, graph_encoder:str="GCN",
# num_gnn_layer:int=2,hidden_sizes:Tuple=(256, 256),
# epoch_num:int=512, max_action:int=512,
# steps_per_epoch=2048, model_path=None

class RL(object):
    def __init__(self, chain:Chain, budget:List[int], slo_latency:float,
                 overrun_lim:float, model_path:str, mode:str, threads:int,
                 # hyperparameters
                 hidden_sizes:Tuple, num_gnn_layer:int, seed:int,
                 steps_per_epoch:int, epochs:int, max_action:int,
                 gamma:float, clip_ratio:float, pi_lr:float, vf_lr:float,
                 train_pi_iters:int, train_v_iters:int, lam:float,
                 max_ep_len:int, target_kl:float, save_freq:int):
        
        # Environment Parameters
        self.chain = chain
        self.budget = budget
        self.slo_latency = slo_latency
        self.overrun_lim = overrun_lim
        self.mode = mode

        # Model Hyperparameters
        self.num_gnn_layer = num_gnn_layer
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.model_path = model_path
        self.max_action = max_action
        self.seed = seed
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.target_kl = target_kl
        self.save_freq = save_freq
        self.max_ep_len = max_ep_len
        
        log_dir_name_list = [int(time.time()), len(self.chain.components),
                             self.steps_per_epoch]
        self.log_dir = '_'.join([str(i) for i in log_dir_name_list])
        self.envs = []
        self.threads = threads


    def get_env(self):
        self.envs.append(CustomEnv(self.log_dir, self.steps_per_epoch,
                                   self.budget, self.slo_latency,
                                   self.overrun_lim, self.mode))
        return self.envs[-1]


    def train(self, algo):
        logger_kwargs = dict(exp_name = "test", seed = self.seed)
        ac_kwargs = dict(graph_encoder_hidden=256,
                         hidden_sizes=self.hidden_sizes,
                         num_gnn_layer=self.num_gnn_layer)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ac = GCNActorCritic

        mpi_fork(self.threads)

        if algo == "ppo":
            ppo(env_fn=self.get_env, ac=ac, ac_kwargs=ac_kwargs,
                seed=self.seed, steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs, gamma=self.gamma,
                clip_ratio=self.clip_ratio, pi_lr=self.pi_lr,
                vf_lr=self.vf_lr, train_pi_iters=self.train_pi_iters,
                train_v_iters=self.train_v_iters, lam=self.lam,
                max_ep_len=self.max_ep_len, target_kl=self.target_kl,
                logger_kwargs=logger_kwargs, save_freq=self.save_freq
                )

        for env in self.envs:
            env:CustomEnv
            env.terminate()
