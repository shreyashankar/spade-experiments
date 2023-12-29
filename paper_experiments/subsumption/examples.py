import os
import pandas as pd

# Get current file's parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Get dataset directory
dataset_dir = os.path.join(parent_dir, "data")

# Recursively get all csv files in the dataset directory
csv_files = []
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

# Read all csv files into dataframes
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
df = pd.concat(dfs)


NUM_EXAMPLES = 50

# Sample 25 true and 25 false examples
example_df = pd.concat(
    [
        df[df["result"] == True].sample(n=NUM_EXAMPLES // 2, random_state=20),
        df[df["result"] == False].sample(n=NUM_EXAMPLES // 2, random_state=20),
    ]
)

EXAMPLES = example_df.to_dict("records")
