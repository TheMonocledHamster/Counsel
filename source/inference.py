import argparse
from utils.test_policy import load_policy_and_env, run_policy

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_path", type=str)
parser.add_argument("-n","--episodes", type=int, default=50)
args = parser.parse_args()

env, get_action = load_policy_and_env(args.model_path, 'last')

run_policy(env, get_action, num_episodes=args.episodes)

