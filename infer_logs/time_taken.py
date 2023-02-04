import os
import pandas as pd
import numpy as np

NCONFS = [5, 10, 25, 50, 100]
NCOMPS = [3, 5, 10, 20]
dir = os.path.dirname(__file__)

for name in ['op','std','up']:
    tracker = pd.read_csv(dir+f'/time_tracker_{name}.csv')

    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.csv') and file.startswith('f'):
                df = pd.read_csv(root+'/'+file, index_col=0)
                tracker['Time/Step'] = tracker['Time'] / df['steps'].sum()

    checker = tracker[['Configs','Components']].values.tolist()


    for nconf in NCONFS:
        for ncomp in NCOMPS:
            pair = [nconf,ncomp]
            if pair not in checker:
                infer_time = tracker[(tracker['Configs'] == nconf) & 
                                        (tracker['Components']==3)]['Time/Step'] * \
                                tracker[(tracker['Configs'] == 5) & (tracker['Components']==ncomp)]['Time/Step'] / \
                                tracker[(tracker['Configs'] == 5) & (tracker['Components']==3)]['Time/Step']
                tracker.append({'Name':name, 'Configs':nconf, 'Components':ncomp, 
                                            'Time':np.nan, 'Time/Step':infer_time}, ignore_index=True)

    print(tracker)
    tracker.to_csv(dir+f'/infer_time_{name}.csv')
