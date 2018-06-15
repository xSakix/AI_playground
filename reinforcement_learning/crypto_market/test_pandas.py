import sys

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

start_date = '1993-01-01'
end_date = '2018-01-01'
prefix = 'btc_'
df_adj_close = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)

data = df_adj_close[['BTC-USD']]
pct = data.iloc[0:30].pct_change().as_matrix()
pct2 = data.pct_change().as_matrix()

print(np.array_equiv(pct, pct2[0:30]))

_, (ax0, ax1) = plt.subplots(1, 2)
ax0.plot(pct)
ax1.plot(pct2[0:30])
# plt.legend(['pct','pct2'])
plt.show()

data2 = data.iloc[0:60]
print(len(data2))
plt.plot(data2)
plt.show()


pct = data2.pct_change().as_matrix()
_, (ax0, ax1) = plt.subplots(1, 2)
ax0.plot(pct)
ax1.plot(pct2[0:60])
plt.show()

pd = pd.DataFrame(np.column_stack((pct,pct2[0:60])))
print(pd)
