from reinforcement_learning.crypto_market.crypto_random_agent import CryptoRandomAgent

import sys

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file
from etf_data_loader import load_all_data_from_file2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns


def gen_random_date(year_low, year_high):
    y = np.random.randint(year_low, year_high)
    m = np.random.randint(1, 12)
    d = np.random.randint(1, 28)
    return datetime(year=y, month=m, day=d)


def get_data_random_dates(df_adj_close, min_year, max_year):
    rand_start = gen_random_date(min_year, max_year)
    rand_end = gen_random_date(min_year, max_year)
    if rand_start > rand_end:
        tmp = rand_start
        rand_start = rand_end
        rand_end = tmp
    data = df_adj_close[df_adj_close['date'] > str(rand_start)]
    data = data[data['date'] < str(rand_end)]

    return data


def clean_data(df_adj_close, ticket):
    top = df_adj_close.index.max()

    for index in df_adj_close.index:
        if df_adj_close.loc[index, ticket] == 0.:
            for i in range(index, top + 1):
                if df_adj_close.loc[i, ticket] > 0.:
                    df_adj_close.loc[index, ticket] = df_adj_close.loc[i, ticket]
                    break
    return df_adj_close


# start_date = '1993-01-01'
# start_date = '2011-09-12'
# end_date = '2018-01-01'
start_date = '2017-10-01'
end_date = '2018-05-18'
prefix = 'btc_'
ticket = 'BTC-USD'

df_adj_close = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)
data1 = df_adj_close
data1 = data1[[ticket]]
data1 = data1.reset_index(drop=True)
print(data1.head())

df = load_all_data_from_file('btc_data_open.csv', start_date, end_date)
# print(df.head())

_,(ax0,ax1) = plt.subplots(2,1)
ax0.plot(df['Open'].ewm(alpha=0.1).mean().pct_change())
ax0.set_title('quandl')
ax1.plot(data1.ewm(alpha=0.1).mean().pct_change())
ax1.set_title('yahoo')
plt.show()

start_date = min(df['Date'])
print(start_date, ' - ', end_date)
data = pd.DataFrame(df['Open'])
print(data.head())

data = data.ewm(alpha=0.1).mean()
data1 = data1.ewm(alpha=0.1).mean()

plt.plot(data.pct_change().cumsum())
plt.plot(data1.pct_change().cumsum())
plt.legend(['quandl','yahoo'])
plt.show()

np.warnings.filterwarnings('ignore')

iter = 0
scores = []

max = None

found = {}


while len(found) < 10:
    if len(data) < 30:
        continue
    agent = CryptoRandomAgent(ticket)
    agent.invest(data, window=30)
    scores.append(agent.score)

    if max is None:
        max = agent
    if max.score < agent.score:
        max = agent

    if agent.score > 0:
        found[agent.score] = agent

    print('\r %d : %d : %.2f : %d : %s' % (iter, agent.score, agent.ror_history[-1], len(found), ticket), end='')
    iter += 1

print('\n scores:', found.keys())

x_train = None
y_train = None
for score in found.keys():
    # print(found[score].state.shape)
    if x_train is None:
        x_train = found[score].state
    else:
        x_train = np.concatenate((x_train, found[score].state))
    if y_train is None:
        y_train = found[score].r_actions
    else:
        y_train = np.concatenate((y_train, found[score].r_actions))

# print(x_train.shape)

x = x_train
print(x.shape)
y = y_train
print(y.shape)

np.save('x.npy', x)
np.save('y.npy', y)

df = pd.DataFrame(columns=['score', 'number_of_actions', 'ror'])

for key in list(found.keys()):
    num = found[key].actions.count('S') + found[key].actions.count('B')
    ror = found[key].ror_history[-1]
    df = df.append({'score': key, 'number_of_actions': num, 'ror': ror}, ignore_index=True)

print(df.sort_values(by='ror', ascending=False))
# print('median actions = ', df['number_of_actions'].median())
#
# m = int(df['score'].loc[0])
# print('\r chosen key:', m, end='')
# while True:
#     try:
#         median_chaos = found[m]
#         break
#     except:
#         if m < 0:
#             exit(1)
#         m -= 1
#         print('\r chosen key:', m, end='')
#
# agent = max
# print('\n Max score:', max.score)
#
# sns.kdeplot(scores)
# plt.show()
#
# plt.plot(agent.ror_history)
# plt.plot(median_chaos.ror_history)
# plt.plot(data[[ticket]].pct_change().cumsum().as_matrix())
# plt.legend(['ror', 'median chaos ror', 'benchmark'])
# plt.title('chaos results of 2/3 of data')
# plt.show()
#
# chaos_counts = [agent.actions.count('S'),
#                 median_chaos.actions.count('S'),
#                 agent.actions.count('B'),
#                 median_chaos.actions.count('B'),
#                 agent.actions.count('H'),
#                 median_chaos.actions.count('H'), ]
# print('\n[S, Sm, B, Bm, H, Hm]\n', chaos_counts)
# # plt.bar(range(6), chaos_counts, width=1, align='center')
# # plt.xticks(['S', 'Sm', 'B', 'Bm', 'H', 'Hm'])
# # plt.show()
