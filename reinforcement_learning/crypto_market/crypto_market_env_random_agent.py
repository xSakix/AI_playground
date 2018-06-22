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


start_date = '2015-01-01'
# start_date = '2018-01-01'
# end_date = '2018-04-01'
end_date = '2018-05-01'
prefix = 'btc_'
ticket = 'LTC-BTC'
dir = 'data_ltc_btc/'

df_adj_close = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)
data = df_adj_close
data = data[[ticket]]
data = data.reset_index(drop=True)
print(data.head())

plt.plot(data)
plt.title(ticket+' pct')
plt.show()

plt.plot(data.pct_change())
plt.title(ticket+' pct')
plt.show()

plt.plot(data.pct_change().cumsum())
plt.legend(['b&h '+ticket])
plt.show()

# exit(1)

np.warnings.filterwarnings('ignore')

iter = 0
scores = []

max = None

found = {}


while len(found) < 20:
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


np.save(dir+'x.npy', x)
np.save(dir+'y.npy', y)

df = pd.DataFrame(columns=['score', 'number_of_actions', 'ror'])

for key in list(found.keys()):
    num = found[key].actions.count('S') + found[key].actions.count('B')
    ror = found[key].ror_history[-1]
    df = df.append({'score': key, 'number_of_actions': num, 'ror': ror}, ignore_index=True)

print(df.sort_values(by='ror', ascending=False))
