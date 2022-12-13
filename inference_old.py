import os
import time
import csv
import numpy as np

from source.utils.test_policy import load_policy_and_env, run_policy
from source.synthetic import set_slo

NCONFS = [5, 10, 25, 50, 100]
NCOMPS = [3, 5, 10, 20]
EPISODE_COUNT = 20

slo = 150
freq = int(1e6 / np.random.randint(int(slo*0.8), int(slo*1.2)))
knob = 0.05
print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")
set_slo(slo, freq, knob)

time_tracker = []

dir = '/home/adi/Work/RoboConf/source/data/'
out_dir = '/home/adi/Work/RoboConf/source/inf_logs/'
out_path = os.path.join(out_dir, 'inference.log')
fix = lambda x: x + '/' + x + '_s0/'

for ncomp in NCOMPS:
    name = f'std-f5-c{ncomp}'
    model_path = os.path.join(dir, fix(name))
    os.makedirs(out_dir+name, exist_ok=True)

    params = {
        'log_dir':out_dir+name,
        'steps_per_epoch': 1000,
        'budget': [1200, 1800],
        'slo_latency': slo,
        'overrun_lim': 0.2,
        'mode': 'synthetic',
        'nconf': 5,
        'ncomp': ncomp,
    }

    env, get_action = load_policy_and_env(model_path, 'last', params)

    start = time.time()
    run_policy(env, get_action, out_path, num_episodes=EPISODE_COUNT)
    end = time.time()

    time_tracker.append([name, end - start])

for nconf in NCONFS:
    name = f'std-f{nconf}-c3'
    model_path = os.path.join(dir, fix(name))

    params = {
        'log_dir':out_dir+name,
        'steps_per_epoch': 1000,
        'budget': [600, 900],
        'slo_latency': slo,
        'overrun_lim': 0.2,
        'mode': 'synthetic',
        'nconf': nconf,
        'ncomp': 3,
    }

    env, get_action = load_policy_and_env(model_path, 'last', params)
    with open(out_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['---------------------'])
        writer.writerow([name])

    start = time.time()
    run_policy(env, get_action, out_path, num_episodes=EPISODE_COUNT)
    end = time.time()

    time_tracker.append([name, end - start])

with open('source/inf_logs/time_tracker.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Time'])
    writer.writerows(time_tracker)
