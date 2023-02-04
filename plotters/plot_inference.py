import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from cycler import cycler


save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'charts/')
NCONFS = [5, 10, 25, 50, 100]
NCOMPS = [3, 5, 10, 20]


dirs = [[f'std-f5-c{NCOMPS}/' for NCOMPS in NCOMPS],
        [f'std-f{NCONF}-c3/' for NCONF in NCONFS]]
titles = ['Varying Number of Components', 'Varying Number of Configurations']
dfs = [[],[]]
labels = [[f'Components = {ncomp}' for ncomp in NCOMPS], 
          [f'Configurations = {nconf}' for nconf in NCONFS]]

# Read CSV file
def read_mod(load_dir, ncomp):
    load_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'inf_logs/', load_dir)
    root, dirs, files = next(os.walk(load_dir))
    csv_file = files[0]
    df = pd.read_csv(load_dir+csv_file, index_col=0)
    df['util'] = gaussian_filter1d(df['episode_reward'] * 100 / (df['steps'] * ncomp), sigma=4)
    return df

for i, dir in enumerate(dirs[0]):
    dfs[0].append(read_mod(dir, NCOMPS[i]))
for dir in dirs[1]:
    dfs[1].append(read_mod(dir, 3))


aspect = 1
n = 1  # number of rows
m = 2  # numberof columns
bottom = 0.1
left = 0.05
top = 1. - bottom
right = 1. - 0.18
fisasp = (1 - bottom - (1 - top))/float(1 - left - (1 - right))
#   widthspace, relative to subplot size
wspace = 0.05  # set to zero for no spacing
hspace = wspace/float(aspect)
#   fix the figure height
figheight = 4  # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp
plt.rc('axes', prop_cycle=(cycler('color', ['red', 'purple', 'orange', 'magenta', 'cyan']) + cycler(
        'linestyle', ['solid', 'dotted', 'dashdot','solid', 'dashed'])))

fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                wspace=wspace, hspace=hspace)

xTicks = [0, 10, 20, 30, 40, 50]

for i in range(len(dfs)):
    axes[i].set_xticks(xTicks)
    axes[i].set_ylim([0, 100])
    axes[i].set_xlim([0, 50])
    axes[i].set_xlabel('Episodes')
    axes[i].set_ylabel('Average Resource Utilization %')
    for df in dfs[i]:
        axes[i].plot(df.index, df['util'])
    axes[i].grid(True)
    axes[i].set_title(titles[i])
    axes[i].legend(labels[i], loc = 'upper left', fontsize = 'x-small')

plt.tight_layout()
plt.savefig(save_dir + 'inference.pdf')
plt.show()
