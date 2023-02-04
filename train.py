import json
import os
import numpy as np
from model.synthetic import set_slo
from model.rl import RL
import argparse


hp_file = "model/configs/hyperparams.json"
hp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), hp_file)
hyperparams = json.load(open(hp_file, "r"))


parser = argparse.ArgumentParser()
parser.add_argument('-n',"--exp_name", help="exp name", type=str)
parser.add_argument('-e',"--epochs", help="epochs", type=int)
parser.add_argument('-c',"--clip_ratio", help="clip ratio (epsilon)", type=float)
parser.add_argument('-k', "--knob", help="knob", type=float)
parser.add_argument('-ncp', "--ncomp", help="NFV Components", type=int)
parser.add_argument('-ncf', "--nconf", help="VM Instances", type=int)
args = parser.parse_args()

if args.exp_name:
    hyperparams["exp_name"] = args.exp_name
if args.clip_ratio:
    hyperparams["clip_ratio"] = float(args.clip_ratio)
if args.knob:
    hyperparams["knob"] = float(args.knob)
if args.ncomp:
    hyperparams["ncomp"] = int(args.ncomp)
if args.nconf:
    hyperparams["nconf"] = int(args.nconf)
if args.epochs:
    hyperparams["epochs"] = int(args.epochs)


slo = int(np.exp(np.random.randint(240,840)/100))
freq = int(1e6 / np.random.randint(int(slo*0.8), int(slo*1.2)))
knob = hyperparams["knob"] # For over, under and near provisioning
print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")
set_slo(slo, freq, knob)

budget = hyperparams["budget"]
overrun_lim = hyperparams["budget_relax"]
mode = hyperparams["mode"]
threads = hyperparams["threads"]
model_path = hyperparams["model_path"]
nconf = hyperparams["nconf"]
ncomp = hyperparams["ncomp"]
algo = hyperparams["algo"]

roboconf = RL(slo=slo, budget=budget, overrun_lim=overrun_lim, 
              mode=mode, threads=threads, model_path=model_path, 
              nconf=nconf, ncomp=ncomp,
              exp_name=hyperparams["exp_name"], 
              hidden_sizes=hyperparams["hidden_sizes"], 
              num_gnn_layer=hyperparams["num_gnn_layer"], 
              seed=hyperparams["seed"], 
              steps_per_epoch=hyperparams["steps_per_epoch"], 
              epochs=hyperparams["epochs"], 
              max_action=hyperparams["max_action"], 
              gamma=hyperparams["gamma"], 
              clip_ratio=hyperparams["clip_ratio"], 
              pi_lr=hyperparams["pi_lr"], 
              vf_lr=hyperparams["vf_lr"], 
              train_pi_iters=hyperparams["train_pi_iters"], 
              train_v_iters=hyperparams["train_v_iters"], 
              lam=hyperparams["lam"], 
              max_ep_len=hyperparams["max_ep_len"], 
              target_kl=hyperparams["target_kl"], 
              save_freq=hyperparams["save_freq"])

roboconf.train(algo)
