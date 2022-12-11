import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import argparse
import matplotlib.pyplot as plt
from cycler import cycler


save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'charts/hp_eval/')

# Read path to CSV file from command line
parser = argparse.ArgumentParser()
parser.add_argument("-e1", "--exp1", type=str)
parser.add_argument("-e2", "--exp2", type=str)
parser.add_argument("-e3", "--exp3", type=str)
args = parser.parse_args()

NCOMPS = 3

fix = lambda x: 'data/' + x + '/' + x + '_s0/'

# dirs = [args.exp1, args.exp2, args.exp3]
dirs = [[fix('op01'), fix('op02'), fix('op03')],
        [fix('up01'), fix('up02'), fix('up03')],
        [fix('std01'), fix('std02'), fix('std03')]]
titles = ['Overprovisioned', 'Underprovisioned', 'Expert']
dfs = [[],[],[]]

# Read CSV file
def read_mod(load_dir):
    load_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), load_dir)
    csv_file = 'progress.csv'
    df = pd.read_csv(load_dir+csv_file, sep='\t', index_col=0)
    df['Reward'] = gaussian_filter1d(df['AverageEpRet'] * 100/ (df['AverageEpLen'] * NCOMPS), sigma=4)
    return df

for i, _dirs in enumerate(dirs):
    for dir in _dirs:
        try:
            dfs[i].append(read_mod(dir))
        except:
            continue

aspect = 1
n = 1  # number of rows
m = 3  # numberof columns
bottom = 0.1
left = 0.05
top = 1. - bottom
right = 1. - 0.18
fisasp = (1 - bottom - (1 - top))/float(1 - left - (1 - right))
#   widthspace, relative to subplot size
wspace = 0.05  # set to zero for no spacing
hspace = wspace/float(aspect)
#   fix the figure height
figheight = 3  # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp
plt.rc('axes', prop_cycle=(cycler('color', ['red', 'magenta', 'orange']) + cycler(
        'linestyle', ['solid', 'dashdot', 'dashed'])))

fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                wspace=wspace, hspace=hspace)

xTicks = [0, 1e6, 2e6, 3e6, 4e6, 5e6]

for i in range(len(dfs)):
    axes[i].set_xticks(xTicks)
    axes[i].set_xlim([0, 5e6])
    axes[i].set_ylim([0, 100])
    axes[i].set_xlabel('Interactions')
    axes[i].set_ylabel('Utilization Reward')
    for df in dfs[i]:
        try:
            axes[i].plot(df['TotalEnvInteracts'], df['Reward'])
        except:
            continue
    axes[i].grid(True)
    axes[i].set_title(titles[i])

plt.tight_layout()
plt.savefig(save_dir + 'hp_eval.pdf')
plt.show()
