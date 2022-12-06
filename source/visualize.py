from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd
import os

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

dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'data/'+exp_name+'/'+exp_name+'_s'+str(seed) + '/')
csv_file = 'progress.csv'

# Read CSV file
df = pd.read_csv(dir+csv_file, sep='\t', index_col=0)

# Calculate average reward
df['Reward'] = df['AverageEpRet'] / (df['AverageEpLen'] * 3)
df['StdReward'] = df['StdEpRet'] / (df['AverageEpLen'] * 3)

sns.set_style("darkgrid")
sns.set_palette("Set1")
fig, ax = plt.subplots(figsize=(20, 10))

middle = gaussian_filter1d(df['Reward'], sigma=3)
upper = gaussian_filter1d(df['Reward'] + df['StdReward'], sigma=3)
lower = gaussian_filter1d(df['Reward'] - df['StdReward'], sigma=3)

# np.clip(lower, 0, 1, out=lower)
# np.clip(upper, 0, 1, out=upper)

ax.plot(df.index, df['Reward'], '--', color='gray', alpha=0.3)
ax.plot(df.index, middle, label='Reward')
# ax.fill_between(df.index, lower, upper, alpha=0.1)


# Output plot to file
# plot.figure.savefig(dir+"reward.png")
plt.title("Reward Per Component vs. Epoch Count")
plt.xlabel("Epoch")
plt.ylabel("Reward")

plt.show()
# plt.savefig(dir+"reward.png")
