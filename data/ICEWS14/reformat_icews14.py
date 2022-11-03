import pandas as pd
import os

cwd = os.getcwd()

print(cwd)

# with open()
train = pd.read_csv(os.path.join(cwd, 'train.txt'), sep='\t', header=None)
valid = pd.read_csv(os.path.join(cwd, 'valid.txt'), sep='\t', header=None)
test = pd.read_csv(os.path.join(cwd, 'test.txt'), sep='\t', header=None)

def reformat(df):
    df[4] = -1
    df[3] = df[3] - 1
    df[3] = df[3] * 24
    return df

train = reformat(train)
valid = reformat(valid)
test = reformat(test)

train.to_csv(os.path.join(cwd, 'icews14s', 'train.txt'), index=None, sep='\t', header=None)
valid.to_csv(os.path.join(cwd, 'icews14s', 'valid.txt'), index=None, sep='\t', header=None)
test.to_csv(os.path.join(cwd, 'icews14s', 'test.txt'), index=None, sep='\t', header=None)
print('done')