from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd
import os

COMP_N = 3
save_dir = os.path.join(os.path.dirname(__file__), 'charts/epoch_wise/')

# Read path to CSV file from command line
parser = argparse.ArgumentParser()
parser.add_argument("-n","--exp_name", help="Name of experiment", type=str)
parser.add_argument("-s", "--seed", help="Seed for experiment", type=int)
args = parser.parse_args()

if args.exp_name is not None:
    exp_name = args.exp_name
else:
    print("Using default experiment name")
    exp_name = "test"

if args.seed is not None:
    seed = args.seed
else:
    print("Using default seed")
    seed = 0

dir = os.path.join(os.path.dirname(__file__),'data/'+
      exp_name+'/'+exp_name+'_s'+str(seed) + '/')
csv_file = 'progress.csv'

# Read CSV file
df = pd.read_csv(dir+csv_file, sep='\t', index_col=0)

# Calculate average reward
df['Reward'] = df['AverageEpRet'] / (df['AverageEpLen'] * COMP_N)
# df['MaxReward'] = df['MaxEpRet'] / (df['MaxEpLen'] * COMP_N)
# df['MinReward'] = df['MinEpRet'] / (df['MinEpLen'] * COMP_N)
# df['StdReward'] = df['StdEpRet'] / (df['AverageEpLen'] * COMP_N)

sns.set_style("darkgrid")
sns.set_palette("Set1")
fig, ax = plt.subplots(figsize=(20, 10))

smooth_mean = gaussian_filter1d(df['Reward'], sigma=3)
# upper = gaussian_filter1d(df['Reward'] + df['StdReward'], sigma=3)
# lower = gaussian_filter1d(df['Reward'] - df['StdReward'], sigma=3)
# max = gaussian_filter1d(df['MaxReward'], sigma=3)
# min = gaussian_filter1d(df['MinReward'], sigma=3)



ax.plot(df['TotalEnvInteracts'], df['Reward'], '--', color='grey', alpha=0.3, linewidth=0.5)
ax.plot(df['TotalEnvInteracts'], smooth_mean, label='Reward', color='red')
# ax.fill_between(df.index, lower, upper, alpha=0.1)
# ax.plot(df.index, max, label='Max Reward', alpha=0.5, linewidth=0.7, color='green')
# ax.plot(df.index, min, label='Min Reward', alpha=0.5, linewidth=0.7, color='red')


plt.title("Resource Utilization vs Time Steps")
plt.xlabel("Steps Taken (1e7)")
plt.ylabel("Resource Utilization")

# plt.show()
plt.savefig(save_dir+f"{exp_name}_{COMP_N}_rewards.png")
