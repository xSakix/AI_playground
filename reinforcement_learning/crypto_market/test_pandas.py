import sys

from sklearn.preprocessing import QuantileTransformer

from reinforcement_learning.crypto_market.crypto_trader_agent import State

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')

# start_date = '2011-08-07'
start_date = '2018-05-01'
# end_date = '2017-12-01'
end_date = '2018-07-23'

df_adj_close = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)
window = 30
data = df_adj_close
data = data[['BTC-EUR']]
data = data.reset_index(drop=True)
data.fillna(method="bfill", inplace=True)

bench = data.pct_change().cumsum().as_matrix()
pct = data.pct_change().as_matrix()
data_1 = pd.DataFrame(pct)
data_1.fillna(method="bfill", inplace=True)

pct = data_1.as_matrix()
mean = data_1.rolling(window=window).mean().as_matrix()
median = data_1.rolling(window=window).median().as_matrix()
std = data_1.rolling(window=window).std().as_matrix()
upperbb = mean + (2 * std)
lowerbb = mean - (2 * std)


plt.plot(pct)
plt.show()

quant = QuantileTransformer()
x = quant.fit_transform(pct)

plt.plot(x)
plt.show()



states = State(window, pct, bench, mean, median, lowerbb, upperbb)

for i in range(window, len(data)):
    input = states.get_state(i)
    print(i, "/", len(data), "->", np.reshape(input, (1, 5)))

# plt.plot(pct)
# plt.plot(mean)
# plt.plot(median)
# plt.plot(lowerbb)
# plt.plot(upperbb)
# plt.show()
