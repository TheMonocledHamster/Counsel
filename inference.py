import os
import time
import csv

from source.utils.test_policy import load_policy_and_env, run_policy
from source.synthetic import set_slo

NCONFS = [5, 10, 25, 50, 100]
NCOMPS = [3, 5, 10, 20]
EPISODE_COUNT = 20

for name, knob in [('op',0.03), ('std',0.05), ('up',0.07)]:
    slo = 150
    freq = int(1e6 / slo)
    print(f"SLO: {slo}, Freq: {freq}, Knob: {knob}")
    set_slo(slo, freq, knob)

    time_tracker = []

    dir = '/home/adi/Work/RoboConf/source/data/'
    out_dir = '/home/adi/Work/RoboConf/source/infer_logs/'
    out_path = os.path.join(out_dir, 'inference.csv')
    fix = lambda x: x + '/' + x + '_s0/'
    
    os.makedirs(out_dir+name, exist_ok=True)
    
    for nconf in NCONFS:
        for ncomp in NCOMPS:
            model_path = os.path.join(dir, fix(f'std-f{nconf}-c{ncomp}'))
            params = {
                'log_dir':out_dir+name,
                'steps_per_epoch': 1000,
                'budget': [2500, 3600],
                'slo_latency': slo,
                'overrun_lim': 0.2,
                'mode': 'synthetic',
                'nconf': nconf,
                'ncomp': ncomp,
            }

            try:
                env, get_action = load_policy_and_env(model_path, 'last', params)
            except:
                continue

            start = time.time()
            run_policy(env, get_action, out_path, num_episodes=EPISODE_COUNT)
            end = time.time()

            time_tracker.append([name, nconf, ncomp, end - start])

    with open(f'source/infer_logs/time_tracker_{name}.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Configs', 'Components', 'Time'])
        writer.writerows(time_tracker)
