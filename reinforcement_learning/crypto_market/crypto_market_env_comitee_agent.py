import os

from reinforcement_learning.crypto_market.comitee_trader_agent import ComiteeTraderAgent
from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent

import sys

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import numpy as np
import matplotlib.pyplot as plt

# start_date = '2010-01-01'
# start_date = '2017-10-01'
start_date = '2017-01-01'
end_date = '2018-06-10'
prefix = 'btc_'
ticket = 'BTC-USD'

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

legends = ['benchmark']

plt.plot(data[[ticket]].pct_change().cumsum().as_matrix())

agent = ComiteeTraderAgent(ticket)
agent.invest(data[[ticket]], window=30)

plt.plot(agent.ror_history)
legends.append('ror comitee')

counts = [agent.actions.count('S'),
          agent.actions.count('B'),
          agent.actions.count('H'), ]
print('\n[S,  B,  H, ]\n', counts)

print('-' * 80)
print(start_date, ' <-> ', end_date)
print('ror:', agent.ror_history[-1])
print('cash:', agent.cash)
print('shares:', agent.shares)
print('value:', agent.history[-1])

models = [
        'decision_tree_4.pkl',
        'decision_tree_9.pkl']


for model in models:
    agent = CryptoTraderAgent(ticket, model='models/'+str(model))
    agent.invest(data[[ticket]], window=30)

    plt.plot(agent.ror_history)
    legends.append('ror '+str(model))

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
