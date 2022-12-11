import argparse
import os
from source.utils.test_policy import load_policy_and_env, run_policy
import time

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_path", type=str)
parser.add_argument("-n","--episodes", type=int, default=50)
args = parser.parse_args()

try:
    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
except:
    model_path = '/home/adi/Work/RoboConf/source/data/std_inf/std_inf_s0/'
env, get_action = load_policy_and_env(model_path, 'last')
env.log_path = model_path+'inference.csv'

start = time.time()
run_policy(env, get_action, num_episodes=args.episodes)
end = time.time()

print('Time taken: ', end-start)
