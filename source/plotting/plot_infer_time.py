import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from cycler import cycler

save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'charts/')
NCONFS = [10, 25, 50, 100]
NCOMPS = [3, 5, 10, 20]

dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'infer_logs/infer_time')
titles = ['Overprovisioned', 'Underprovisioned', 'Expert']

names = ['op','std','up']
idfs  = [[],[],[]]
dfs = [[],[],[]]

for i,name in enumerate(names):
    csv_file = dir + f'_{name}.csv'
    idfs[i] = pd.read_csv(csv_file)
    dfs[i].append([idfs[i][idfs[i]['Configs'] == NCONF][['Configs','Components','Time/Step']] for NCONF in NCONFS])




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
plt.rc('axes', prop_cycle=(cycler('color', ['red', 'purple', 'orange', 'magenta', 'cyan']) + cycler(
        'linestyle', ['solid', 'dotted', 'dashdot','solid', 'dashed'])))

fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                wspace=wspace, hspace=hspace)

xTicks = [3,5,10,20]

for i in range(len(dfs)):
    axes[i].set_xticks(xTicks)
    axes[i].set_ylim([0, 10])
    axes[i].set_xlabel('Components')
    axes[i].set_ylabel('Inference Time (ms)')
    for prov in dfs[i]:
        for df in prov:
            axes[i].plot(df['Components'], df['Time/Step'] * 1000, label = f'Configs = {df["Configs"].iloc[0]}')
    axes[i].grid(True)
    axes[i].set_title(titles[i])
    axes[i].legend(loc = 'upper left', fontsize = 'x-small')

plt.tight_layout()
plt.savefig(save_dir + 'infer_time.pdf')
plt.show()
