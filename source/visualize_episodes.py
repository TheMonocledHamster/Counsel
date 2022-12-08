from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

COMP_N = 3
save_dir = os.path.join(os.path.dirname(__file__), 'charts/episode_wise/')

# Read path to CSV file from command line
parser = argparse.ArgumentParser()
parser.add_argument("-d","--log_dir", type=str)
parser.add_argument("-t","--chart_type", type=str)
args = parser.parse_args()

if args.log_dir is not None:
    log_dir = args.log_dir
else:
    raise ValueError("No log directory specified")
if args.chart_type in ["reward", "steps", "both"]:
    chart_type = args.chart_type
else:
    raise ValueError("No valid chart type specified")

exp_type = os.path.basename(os.path.normpath(log_dir))
exp_type = exp_type.split("_")[0]

# Read CSV file
df = []
for log_file in os.listdir(log_dir):
    if log_file.endswith(".csv"):
        log_file = os.path.join(log_dir, log_file)
        df.append(pd.read_csv(log_file, index_col=0))

colors = ["red", "blue", "green", "orange", "purple", "cyan", "pink",  "magenta", "yellow"]

if chart_type == "reward" or chart_type == "both":
    fig, ax = plt.subplots(figsize=(20, 10))
    for i in range(len(df)):
        smooth_rwd = gaussian_filter1d(df[i]['episode_reward']/(df[i]['steps']*COMP_N), sigma=3)
        ax.plot(df[i].index, smooth_rwd, label=f'Thread{i}', color=colors[i])

    plt.title("Resource Utilization vs Episode Counter")
    plt.xlabel("Episode Count")
    plt.ylabel("Resource Utilization")

    # plt.show()
    plt.savefig(os.path.join(save_dir, f"{exp_type}_{COMP_N}_rewards.png"))

if chart_type == "steps" or chart_type == "both":
    fig_s, ax_s = plt.subplots(figsize=(20, 10))
    for i in range(len(df)):
        smooth_steps = gaussian_filter1d(df[i]['steps'], sigma=5)
        ax_s.plot(df[i].index, smooth_steps, label=f'Thread{i}', color=colors[i], linestyle='--')
        smooth_actions = gaussian_filter1d(df[i]['actions'], sigma=5)
        ax_s.plot(df[i].index, smooth_actions, label=f'Thread{i}', color=colors[i])

    plt.title("Steps/Actions vs Episode Counter")
    plt.xlabel("Episode Count")
    plt.ylabel("Steps/Actions")

    # plt.show()
    plt.savefig(os.path.join(save_dir, f"{exp_type}_{COMP_N}_actions_steps.png"))
