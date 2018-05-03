import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas_datareader import data as data_reader

data_file = 'btc.data.csv'

if not os.path.isfile(data_file):
    df = data_reader.get_data_yahoo('BTC-USD')
    df.to_csv(data_file)

df = pd.read_csv(data_file)
df = df.Open
print(df)

NUM_OF_SEQUENCES = 29

dfs = []
cols = []
for i in range(0, NUM_OF_SEQUENCES):
    cols.append(str(i))
    dfs.append(df.shift(-i))

result = pd.concat(dfs, axis=1)
result.columns = cols
result.dropna(inplace=True)
result = result.ewm(alpha=0.55).mean()

print(result)

result.to_csv('btc-seq.csv')

# plt.plot(df)
# plt.plot(result)
# plt.show()
