from reinforcement_learning.market.random_agent import RandomAgent

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
prefix = 'mil_'
ranked = pd.read_csv('../../../buy_hold_simulation/evaluation_results/mil_evaluation_result_1.csv')
tickets = ranked.sort_values(by='bayes_interval_high',ascending=False)['ticket'].tolist()[:10]
print(tickets)
df_adj_close = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)

np.warnings.filterwarnings('ignore')

iter = 0
scores = []

max = None

found = {}


while len(found) < 100:
    ticket = 'ANX.MI'
    data = get_data_random_dates(df_adj_close,1993,2018)
    data = data[[ticket]]
    if len(data) < 30:
        continue

    agent = RandomAgent(ticket)
    agent.invest(data, window=30)
    scores.append(agent.score)
    if max is None:
        max = agent
    if max.score < agent.score:
        max = agent

    if agent.score > 0:
        found[agent.score] = agent

    # print('\r %d : %d : %d : %f : %s' % (iter, agent.score, len(found), frac,ticket), end='')
    print('\r %d : %d : %d : %s' % (iter, agent.score, len(found), ticket), end='')
    iter += 1

print('\n scores:', found.keys())

x_train = None
y_train = None
for score in found.keys():
    # print(found[score].state.shape)
    if x_train is None:
        x_train = found[score].state
    else:
        x_train = np.concatenate((x_train,found[score].state))
    if y_train is None:
        y_train= found[score].r_actions
    else:
        y_train = np.concatenate((y_train,found[score].r_actions))

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
