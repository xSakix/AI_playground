
from reinforcement_learning.market.chaos_sim import RandomAgent

import sys
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
    data = df_adj_close[df_adj_close['Date'] > str(rand_start)]
    data = data[data['Date'] < str(rand_end)]

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
end_date = '2018-04-08'
prefix = 'mil_'
ticket = 'ANX.MI'

df_adj_close = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)

np.warnings.filterwarnings('ignore')

try:
    df_ticket_data = df_adj_close[['date', ticket]]
except:
    print('failed to find ticket: ' + ticket)
    exit(1)
df_ticket_data = df_ticket_data[df_ticket_data[ticket] > 0.]
df_ticket_data = df_ticket_data.reindex(method='bfill')

data = df_ticket_data[:2 * int(len(df_ticket_data) / 3)]
data = data.reindex()
val_data = df_ticket_data[2 * int(len(df_ticket_data) / 3):]
val_data = val_data.reindex()
print(data.head())
print(val_data.head())

_, (ax0, ax1) = plt.subplots(1, 2)
ax0.plot(data[[ticket]])
ax1.plot(val_data[[ticket]])
plt.show()

iter = 0
scores = []

max = None

found = {}

while True:
    chaos = RandomAgent(ticket)
    chaos.invest(data[[ticket]])
    scores.append(chaos.score)
    # if chaos.score > 1229:
    if max is None:
        max = chaos
    if max.score < chaos.score:
        max = chaos

    if chaos.score > 0 and (chaos.actions.count('S') + chaos.actions.count('B')) > 100:
        found[chaos.score] = chaos

    if iter > 200:
        break
    print('\r %d : %d : %s' % (iter, chaos.score, chaos.r), end='')
    iter += 1

print('\n scores:', found.keys())

df = pd.DataFrame(columns=['score', 'number_of_actions', 'ror'])

for key in list(found.keys()):
    num = found[key].actions.count('S') + found[key].actions.count('B')
    ror = found[key].ror_history[-1]
    df = df.append({'score': key, 'number_of_actions': num, 'ror': ror}, ignore_index=True)

print(df.sort_values(by='ror', ascending=False))
print('median actions = ', df['number_of_actions'].median())

m = int(df['score'].loc[0])
print('\r chosen key:', m, end='')
while True:
    try:
        median_chaos = found[m]
        break
    except:
        if m < 0:
            exit(1)
        m -= 1
        print('\r chosen key:', m, end='')

chaos = max
print('\n Max score:', max.score)

sns.kdeplot(scores)
plt.show()

chaos2 = RandomAgent(ticket, r=chaos.r)
chaos2.invest(val_data[[ticket]])

median_chaos2 = RandomAgent(ticket, r=median_chaos.r)
median_chaos2.invest(val_data[[ticket]])

print('\nValidation score: ', chaos2.score)

_, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(chaos.ror_history)
ax1.plot(median_chaos.ror_history)
ax1.plot(data[[ticket]].pct_change().cumsum().as_matrix())
ax1.legend(['ror', 'median chaos ror', 'benchmark'])
ax1.set_title('chaos results of 2/3 of data')
ax2.plot(chaos2.ror_history)
ax2.plot(median_chaos2.ror_history)
ax2.plot(val_data[[ticket]].pct_change().cumsum().as_matrix())
ax2.set_title('chaos results of val data')
ax2.legend(['ror', 'median chaos ror', 'benchmark'])
plt.show()

_, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.plot(chaos.x0_history[:20])
ax1.plot(chaos.x1_history[:20])
ax2.plot(chaos.x2_history[:20])
ax0.plot(median_chaos.x0_history[:20])
ax1.plot(median_chaos.x1_history[:20])
ax2.plot(median_chaos.x2_history[:20])
ax0.legend(['S', 'Sm'])
ax1.legend(['B', 'Bm'])
ax2.legend(['H', 'Hm'])
plt.show()

chaos_counts = [chaos.actions.count('S'),
                median_chaos.actions.count('S'),
                chaos.actions.count('B'),
                median_chaos.actions.count('B'),
                chaos.actions.count('H'),
                median_chaos.actions.count('H'), ]
print('\n[S, Sm, B, Bm, H, Hm]\n', chaos_counts)
chaos2_counts = [chaos2.actions.count('S'),
                median_chaos2.actions.count('S'),
                chaos2.actions.count('B'),
                median_chaos2.actions.count('B'),
                chaos2.actions.count('H'),
                median_chaos2.actions.count('H'), ]
print('\n[S, Sm, B, Bm, H, Hm]\n', chaos2_counts)
_, (ax0, ax1) = plt.subplots(2, 1)
ax0.bar(range(6), chaos_counts, width=1, align='center')
ax0.set_xticklabels(['S', 'Sm', 'B', 'Bm', 'H', 'Hm'])
ax1.bar(range(6), chaos2_counts, width=1, align='center')
ax1.set_xticklabels(['S', 'Sm', 'B', 'Bm', 'H', 'Hm'])
plt.show()
