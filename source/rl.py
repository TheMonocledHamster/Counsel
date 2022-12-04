import time
from typing import Tuple, List

from .env import CloudEnv
from .core import GCNActorCritic
from .ppo import ppo
from .utils.mpi_tools import mpi_fork
from .utils.run_utils import setup_logger_kwargs


class RL(object):
    def __init__(self, slo:float, budget:List[int], overrun_lim:float, 
                 mode:str, threads:int, model_path:str,
                 # hyperparameters
                 exp_name:str, hidden_sizes:Tuple, num_gnn_layer:int,
                 seed:int, steps_per_epoch:int, epochs:int, max_action:int,
                 gamma:float, clip_ratio:float, pi_lr:float, vf_lr:float,
                 train_pi_iters:int, train_v_iters:int, lam:float,
                 max_ep_len:int, target_kl:float, save_freq:int):
        
        # Environment Parameters
        self.slo = slo
        self.budget = budget
        self.overrun_lim = overrun_lim
        self.mode = mode

        # Model Hyperparameters
        self.num_gnn_layer = num_gnn_layer
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
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
        self.exp_name = exp_name
        
        log_dir_name_list = [int(time.time()),self.steps_per_epoch]
        self.log_dir = '_'.join([str(i) for i in log_dir_name_list])
        self.envs = []
        self.threads = threads


    def get_env(self):
        self.envs.append(CloudEnv(self.log_dir, self.steps_per_epoch,
                                  self.budget, self.slo,
                                  self.overrun_lim, self.mode))
        return self.envs[-1]


    def train(self, algo):
        logger_kwargs = setup_logger_kwargs(self.exp_name,self.seed)
        ac_kwargs = dict(graph_encoder_hidden=256,
                         hidden_sizes=self.hidden_sizes,
                         num_gnn_layer=self.num_gnn_layer)
        ac = GCNActorCritic

        mpi_fork(self.threads, bind_to_core=True)

        if algo == "ppo":
            ppo(env_fn=self.get_env, actor_critic=ac, ac_kwargs=ac_kwargs,
                seed=self.seed, steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs, gamma=self.gamma,
                clip_ratio=self.clip_ratio, pi_lr=self.pi_lr,
                vf_lr=self.vf_lr, train_pi_iters=self.train_pi_iters,
                train_v_iters=self.train_v_iters, lam=self.lam,
                max_ep_len=self.max_ep_len, target_kl=self.target_kl,
                logger_kwargs=logger_kwargs, save_freq=self.save_freq
                )

        for env in self.envs:
            env:CloudEnv
            env.terminate()
