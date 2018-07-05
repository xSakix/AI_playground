import os
from datetime import datetime

from sklearn.preprocessing import LabelBinarizer
from reinforcement_learning.crypto_market.crypto_random_agent import CryptoRandomAgent
import sys

from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import numpy as np
import matplotlib.pyplot as plt

from choose_best_score_agent import find_best_models
import pandas as pd


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

    return data, rand_start, rand_end


dir = 'data_btc_eur/'

x_file = dir + 'x.npy'
if os.path.isfile(x_file):
    print(x_file, ' removing...')
    os.remove(x_file)

y_file = dir + 'y.npy'
if os.path.isfile(y_file):
    print(y_file, ' removing...')
    os.remove(y_file)

start_date = '2011-08-07'
end_date = '2018-06-27'
prefix = 'btc_'
ticket = 'BTC-EUR'
window = 30

df_adj_close = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)

it = 0

while it < 100:
    print('Iteration:', it)
    data, new_start_date, new_end_date = get_data_random_dates(df_adj_close, 2011, 2018)

    while len(data) < 30:
        data, new_start_date, new_end_date = get_data_random_dates(df_adj_close, 2011, 2018)

    print(new_start_date, ' - ', new_end_date)
    data = data[[ticket]]
    data = data.reset_index(drop=True)
    data.fillna(method="bfill", inplace=True)
    print(data.head(2))
    print(data.tail(2))

    pct = data.pct_change().as_matrix()
    bench = data.pct_change().cumsum().as_matrix()
    data_1 = pd.DataFrame(pct)
    mean = data_1.rolling(window=window).mean().as_matrix()
    median = data_1.rolling(window=window).median().as_matrix()
    std = data_1.rolling(window=window).std().as_matrix()
    upperbb = mean + (2 * std)
    lowerbb = mean - (2 * std)

    np.warnings.filterwarnings('ignore')

    iter = 0
    scores = []

    best = None

    found = {}

    _, clf_agent, _ = find_best_models(start_date=str(new_start_date), end_date=str(new_end_date),
                                       dir_models='best_models_btc_eur/')
    print('Best score agent(%s - %s) is %s' % (new_start_date, new_end_date, clf_agent.model))

    MAX_POP = 100
    COPY_SEL = 2
    SELECTION = 50
    MAX_ITER = 20

    MUTATION_RATE = 0.7

    pop = [CryptoRandomAgent(ticket, use_trader=True, agent=clf_agent) for _ in range(MAX_POP)]

    last_best = None
    last_5 = []
    while iter < MAX_ITER:
        if len(data) < 30:
            continue

        for agent in pop:
            agent.invest(data,
                         window=30,
                         pct=pct,
                         bench=bench,
                         mean=mean,
                         median=median,
                         lowerbb=lowerbb,
                         upperbb=upperbb)

        scores = [agent.score for agent in pop]
        sor = np.argsort(scores)
        idx = sor[::-1]
        print(pop[sor[0]].score, ' <> ', pop[sor[len(pop) - 1]].score)
        best = pop[idx[0]]
        score = pop[idx[0]].score
        last_5.append(score)

        if len(last_5) == 5 and np.sum(np.array(last_5) - last_5[0]) == 0.:
            print('Early stopping criterion reached. Stopping....')
            last_5 = []
            break

        newpop = []
        for index in idx[:COPY_SEL]:
            newpop.append(CryptoRandomAgent(ticket, r_actions=pop[index].r_actions, use_trader=True, agent=clf_agent))

        add = 0
        if last_best is not None and best.score - last_best.score == 0 and MUTATION_RATE >= 0.1:
            MUTATION_RATE -= 0.01
        elif MUTATION_RATE <= 0.9:
            MUTATION_RATE += 0.01

        while add < MAX_POP - COPY_SEL:
            idx1 = np.random.randint(0, SELECTION)
            idx2 = np.random.randint(0, SELECTION)
            a = pop[idx[idx1]].r_actions.copy()
            b = pop[idx[idx2]].r_actions.copy()

            # for _ in range(2):
            index = np.random.randint(0, len(a))
            tmp = a[:index]
            tmp2 = b[:index]
            a[:index] = tmp2
            b[:index] = tmp

            if np.random.normal(0., 1.) < MUTATION_RATE:
                mut_size = int(len(pop) * 0.1)
                indexes = np.random.randint(0, len(a), size=mut_size)
                a[indexes] = np.random.randint(0, 3, size=mut_size)
                b[indexes] = np.random.randint(0, 3, size=mut_size)

            # lb = LabelBinarizer()
            # lb1 = LabelBinarizer()
            #
            # aa = lb.fit_transform(a)
            # bb = lb1.fit_transform(b)
            #
            # id_cross = np.random.randint(0, len(aa[0]))
            # tmp = aa[:, id_cross]
            # tmp2 = bb[:, id_cross]
            #
            # aa[:, id_cross] = tmp2
            # bb[:, id_cross] = tmp
            #
            # a = lb.inverse_transform(aa)
            # b = lb1.inverse_transform(bb)

            newpop.extend([CryptoRandomAgent(ticket, r_actions=a, use_trader=True, agent=clf_agent),
                           CryptoRandomAgent(ticket, r_actions=b, use_trader=True, agent=clf_agent)])
            add += 2
        pop = newpop
        # print('\r %d : %d : %.2f : %d : %s' % (iter, agent.score, agent.ror_history[-1], len(found), ticket), end='')
        print(iter, '->', score, ' | ', len(pop), ',', MUTATION_RATE)
        iter += 1
        last_best = best

    print('\n best score:', best.score)
    print('ror:', best.ror_history[-1])
    if best.score < 0:
        continue
    x = best.state
    y = best.r_actions
    print(x.shape)
    print(y.shape)
    it += 1

    if os.path.isfile(x_file):
        xx = np.load(x_file)
        x = np.concatenate([x, xx])

    if os.path.isfile(y_file):
        yy = np.load(y_file)
        y = np.concatenate([y, yy])

    np.save(x_file, x)
    np.save(y_file, y)
