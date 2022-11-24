import time
import torch

from .env import CustomEnv
from core import GCNActorCritic
import ppo


class RL(object):
    def __init__(
                self, chain, graph_encoder="GCN", num_gnn_layer=2,
                hidden_sizes=(256, 256),epoch_num=1024, 
                max_action=512, steps_per_epoch=1024,
                model_path=None
                ):
        
        self.chain = chain
        
        self.graph_encoder = graph_encoder
        self.num_gnn_layer = num_gnn_layer
        self.hidden_sizes = hidden_sizes

        self.epoch_num = epoch_num
        self.max_action = max_action
        self.steps_per_epoch = steps_per_epoch


        self.model_path = model_path
        
        log_dir_name_list = [
                            int(time.time()), len(self.chain.components), 
                            self.graph_encoder, self.steps_per_epoch
                            ]
        self.log_dir = '_'.join([str(i) for i in log_dir_name_list])


    def get_env(self):
        self.env = CustomEnv(
                            self.chain, log_dir=self.log_dir,
                            budget=self.budget, 
                            slo_latency=self.slo_latency,
                            alpha_lim=self.alpha_lim,
                            steps_per_epoch=self.steps_per_epoch,
                            )
        return self.env


    def run_training(self, algo):
        logger_kwargs = dict(output_dir="results/{}".format(self.log_dir), 
                             exp_name="test")
        ac_kwargs = dict(graph_encoder_hidden=256, 
                         hidden_sizes=self.hidden_sizes, 
                         num_gnn_layer=self.num_gnn_layer)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ac = GCNActorCritic

        # TODO verify hyperparameters
        ppo(self.get_env, enable_mpi=False, non_blocking=False, gamma=0.98,
            actor_critic=ac, max_ep_len=self.max_action, seed=8,
            clip_ratio=0.1, device=device, model_path=self.model_path,
            ac_kwargs=ac_kwargs, epochs=self.epoch_num,
            steps_per_epoch=self.steps_per_epoch,
            logger_kwargs=logger_kwargs)

        self.env.terminate()
