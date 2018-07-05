import os

from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent

import sys

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import numpy as np
import matplotlib.pyplot as plt

dir_data = 'data_btc_eur/'
dir_models = 'models_btc_eur/'
ticket = 'BTC-EUR'
# start_date = '2010-01-01'
# start_date = '2017-10-01'
start_date = '2017-01-01'
end_date = '2018-06-15'
prefix = 'btc_'

df_adj_close = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)

np.warnings.filterwarnings('ignore')

try:
    df_ticket_data = df_adj_close[['date', ticket]]
except:
    print('failed to find ticket: ' + ticket)
    exit(1)
df_ticket_data = df_ticket_data[df_ticket_data[ticket] > 0.]
df_ticket_data = df_ticket_data.reindex(method='bfill')

print(df_ticket_data.head())
print(df_ticket_data.tail())

plt.plot(df_ticket_data[[ticket]])
plt.show()

data = df_ticket_data
models = sorted(os.listdir(dir_models))

legends = ['benchmark']

plt.plot(data[[ticket]].pct_change().cumsum().as_matrix())

count = 0
max = -1.
max_agent = None

for model in models:
    print(model)
    agent = CryptoTraderAgent(ticket, model=dir_models+str(model), binarizer='keras_model_eu/label_bin.pkl')
    agent.invest(data[[ticket]], window=30)

    plt.plot(agent.ror_history)
    legends.append('ror ' + str(model))

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

    if max < agent.ror_history[-1]:
        max =agent.ror_history[-1]
        max_agent = agent

    count += 1

    if count == len(models) or count == 3:
        plt.legend(legends)
        plt.show()
        count = 0
        legends = ['benchmark']

        plt.plot(data[[ticket]].pct_change().cumsum().as_matrix())

legends.append(max_agent.model)
plt.plot(max_agent.ror_history)
plt.legend(legends)
plt.show()
