import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from cycler import cycler


save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'charts/')

# parser = argparse.ArgumentParser()
# parser.add_argument("-e1", "--exp1", type=str)
# parser.add_argument("-e2", "--exp2", type=str)
# parser.add_argument("-e3", "--exp3", type=str)
# args = parser.parse_args()

fix = lambda x: 'data/' + x + '/' + x + '_s0/'

# dirs = [args.exp1, args.exp2, args.exp3]
dirs = [fix('op'), fix('up'), fix('std')]
title = 'Initial Policy'
labels = ['Overprovisioned', 'Underprovisioned', 'Expert']
dfs = []

# Read CSV file
def read_mod(load_dir):
    load_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), load_dir)
    csv_file = 'progress.csv'
    df = pd.read_csv(load_dir+csv_file, sep='\t', index_col=0)
    df['Reward'] = df['AverageEpRet'] / (df['AverageEpLen'] * 3)
    return df

for dir in dirs:
    dfs.append(read_mod(dir))


aspect = 1
n = 1  # number of rows
m = 1  # numberof columns
bottom = 0.1
left = 0.05
top = 1. - bottom
right = 1. - 0.18
fisasp = (1 - bottom - (1 - top))/float(1 - left - (1 - right))
#   widthspace, relative to subplot size
wspace = 0.05  # set to zero for no spacing
hspace = wspace/float(aspect)
#   fix the figure height
figheight = 5  # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp
plt.rc('axes', prop_cycle=(cycler('color', ['red', 'magenta', 'orange', 'green', 'yellow']) + cycler(
        'linestyle', ['solid', 'dashed', 'dashdot','dotted', 'solid'])))

fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                wspace=wspace, hspace=hspace)

xTicks = [0, 2e6, 4e6, 6e6, 8e6, 10e6]

axes.set_xticks(xTicks)
axes.set_ylim([0, 1])
axes.set_xlabel('Training Steps')
axes.set_ylabel('Average Reward/Component')
for i in range(len(dfs)):
    axes.plot(dfs[i]['TotalEnvInteracts'], dfs[i]['Reward'], label=labels[i])
axes.grid(True)
axes.legend(loc='upper left', fontsize='small')

axes.set_title(title)

plt.tight_layout()
plt.savefig(save_dir + 'provisioning.pdf')
plt.show()
