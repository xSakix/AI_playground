import os

from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent

import sys

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dir_data = 'data_ltc_eur/'
dir_models = 'models_ltc_eur/'
ticket = 'LTC_EUR'


start_date = '2017-01-01'
end_date = '2018-06-01'
df_adj_close = load_all_data_from_file('btc_data_open.csv', start_date, end_date)
data = pd.DataFrame(df_adj_close['Open'])
# data = data.ewm(alpha=0.1).mean()

np.warnings.filterwarnings('ignore')

plt.plot(data.pct_change().cumsum().as_matrix())
legends = ['benchmark']


agent = CryptoTraderAgent('btc', model='models/decision_tree_13.pkl')
agent.invest(data, window=30)

plt.plot(agent.ror_history)
legends.append('ror decision_tree_11.pkl')

chaos_counts = [agent.actions.count('S'),
                agent.actions.count('B'),
                agent.actions.count('H'), ]
print('\n[S,  B,  H, ]\n', chaos_counts)

print('-' * 80)
print(start_date, ' <-> ', end_date)
print('ror:', agent.ror_history[-1])
print('cash:', agent.cash)
print('shares:', agent.shares)
print('value:', agent.history[-1])

plt.legend(legends)
plt.show()
