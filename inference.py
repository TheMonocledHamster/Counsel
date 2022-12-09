import argparse
import os
import json
import numpy as np
from source.rl import RL

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_path", type=str)
args = parser.parse_args()

if args.model_path is not None:
    model_path = args.model_path
else:
    raise ValueError("Model path not provided")

params_file = "source/configs/inference_params.json"
params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), params_file)
params = json.load(open(params_file, "r"))
