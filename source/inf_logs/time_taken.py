import os
import pandas as pd
import numpy as np


dir = os.path.dirname(__file__)

tracker = pd.read_csv(dir+'/time_tracker.csv')
tracker['Steps'] = 0

for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith('.csv') and file.startswith('_') and file != 'time_tracker.csv':
            df = pd.read_csv(root+'/'+file, index_col=0)
            dir_name  = root.split('/')[-1]
            tracker.loc[tracker['Name'] == dir_name, 'Steps'] = df['steps'].sum()

tracker['Infer Time per Step'] = tracker['Time'] / tracker['Steps']
tracker.to_csv(dir+'/infer_time.csv')
