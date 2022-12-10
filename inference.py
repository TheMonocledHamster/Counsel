import argparse
import os
from source.utils.test_policy import load_policy_and_env, run_policy

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_path", type=str)
parser.add_argument("-n","--episodes", type=int, default=50)
args = parser.parse_args()

try:
    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
except:
    model_path = '/home/adi/Work/RoboConf/source/data/std/std_s0/'
env, get_action = load_policy_and_env(model_path, 'last')

run_policy(env, get_action, num_episodes=args.episodes)
