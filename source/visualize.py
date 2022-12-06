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

csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'data/'+exp_name+'/'+exp_name+'_s'+str(seed)+'/progress.csv')

# Read CSV file
df = pd.read_csv(csv_file, sep='\t', index_col=0)
print(df.tail())

# Plot Average Step Reward per component vs. Epoch Number
sns.lineplot(data=df, x="Epoch", y="Reward")