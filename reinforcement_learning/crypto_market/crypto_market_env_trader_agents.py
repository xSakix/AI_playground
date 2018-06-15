import os

from reinforcement_learning.crypto_market.crypto_random_agent import CryptoRandomAgent

import sys

from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent

sys.path.insert(0, '../../../etf_data')
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


start_date = '1993-01-01'
end_date = '2018-01-01'
prefix = 'btc_'
df_adj_close = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)

np.warnings.filterwarnings('ignore')

iter = 0
scores = []

max = None

found = {}

ticket = 'BTC-USD'

agents = [CryptoTraderAgent(ticket, model='models/' + model) for model in os.listdir('models')]

# data = get_data_random_dates(df_adj_close, 2010, 2018)
data = df_adj_close
data = data[[ticket]]

for agent in agents:
    agent.invest(data, window=30)

x_train = []
y_train = []

[print(len(agent.ror_history)) for agent in agents]
[print(len(agent.r_actions)) for agent in agents]

counts = np.zeros(len(agents), dtype=np.int32)

for i in range(31, len(data)):
    rors = np.array([agent.ror_history[i] for agent in agents])
    idx = np.argmax(rors)
    counts[idx] += 1
    try:
        x_train.append(agents[idx].state[i - 31])
        y_train.append(agents[idx].r_actions[i - 31])
    except IndexError:
        print(i, ',', i - 31)

x = np.array(x_train)
print(x.shape)
y = np.array(y_train)
print(y.shape)

np.save('x.npy', x)
np.save('y.npy', y)

for idx in range(len(counts)):
    print(idx, ' : ', counts[idx])
