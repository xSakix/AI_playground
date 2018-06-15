from reinforcement_learning.market.trader_agent import TraderAgent

import sys

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import numpy as np
import matplotlib.pyplot as plt

def print_agent_stats(agent):
    print('ror:', agent.ror_history[-1])
    print('cash:', agent.cash)
    print('shares:', agent.shares)
    print('value:', agent.history[-1])


start_date = '1993-01-01'
# start_date = '2015-01-01'
end_date = '2018-06-01'
prefix = 'mil_'
ticket = 'ANX.MI'
# ticket = 'CSNDX.MI'
# ticket = 'EQQQ.MI'
# ticket = 'SPY5.MI'
# ticket = 'CL2.MI'


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

agent1 = TraderAgent(ticket)
agent1.invest(data[[ticket]], window=30)

agent = TraderAgent(ticket, model='../crypto_market/models/decision_tree_4.pkl')
agent.invest(data[[ticket]], window=30)

plt.plot(agent.ror_history)
plt.plot(agent1.ror_history)
plt.plot(data[[ticket]].pct_change().cumsum().as_matrix())
plt.legend(['ror', 'ror orig', 'benchmark'])
plt.title('results')
plt.show()

chaos_counts = [agent.actions.count('S'),
                agent.actions.count('B'),
                agent.actions.count('H'), ]
print('\n[S,  B,  H, ]\n', chaos_counts)

print_agent_stats(agent)
print('-'*80)
print_agent_stats(agent1)

