import os
from datetime import datetime

from sklearn.preprocessing import LabelBinarizer

from reinforcement_learning.crypto_market import crypto_sklearnclass_agent, crypto_bayes_agent
from reinforcement_learning.crypto_market.crypto_random_agent import CryptoRandomAgent, State
import sys

from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import numpy as np
import matplotlib.pyplot as plt

from choose_best_score_agent import find_best_models
import pandas as pd

from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import time

np.warnings.filterwarnings('ignore')

def gen_random_date(year_low, year_high):
    y = np.random.randint(year_low, year_high)
    m = np.random.randint(1, 12)
    d = np.random.randint(1, 28)
    return datetime(year=y, month=m, day=d)


def get_data_random_dates(min_year, max_year):
    rand_start = gen_random_date(min_year, max_year)
    rand_end = gen_random_date(min_year, max_year)
    if rand_start > rand_end:
        tmp = rand_start
        rand_start = rand_end
        rand_end = tmp

    return load_all_data_from_file2('btc_etf_data_adj_close.csv', str(rand_start), str(rand_end)), rand_start, rand_end


start_date = '2017-01-01'
end_date = '2018-07-01'

ticket = 'BTC-EUR'
window = 30

it = 0
dir_data = 'data_ga_periodic'

classifier_scores = []
classifier_rors = []
classifier_acc_scores = []

x_file = dir_data + '/x.npy'
y_file = dir_data + '/y.npy'

if os.path.isfile(x_file):
    print(x_file, ' removing...')
    os.remove(x_file)

if os.path.isfile(y_file):
    print(y_file, ' removing...')
    os.remove(y_file)

while it < 100:
    print('Iteration:', it)

    data, start_date, end_date = get_data_random_dates(2011, 2018)

    while len(data) < 30:
        data, start_date, end_date = get_data_random_dates(2011, 2018)

    print(start_date, ' - ', end_date)
    data = data[[ticket]]
    data = data.reset_index(drop=True)
    data.fillna(method="bfill", inplace=True)
    print(data.head(2))
    print(data.tail(2))
    print(len(data))

    bench = data.pct_change().cumsum().as_matrix()
    pct = data.pct_change().as_matrix()
    data_1 = pd.DataFrame(pct)
    mean = data_1.rolling(window=window).mean().as_matrix()
    median = data_1.rolling(window=window).median().as_matrix()
    std = data_1.rolling(window=window).std().as_matrix()
    upperbb = mean + (2 * std)
    lowerbb = mean - (2 * std)

    states = State(window, pct, bench, mean, median, lowerbb, upperbb)

    iter = 0
    scores = []

    best = None

    found = {}

    MAX_POP = 20
    COPY_SEL = 2
    SELECTION = 10
    MAX_ITER = 100

    MUTATION_RATE = 0.7

    pop = [CryptoRandomAgent(ticket, use_trader=False) for _ in range(MAX_POP)]

    last_best = None
    last_5 = []
    while True:
        if len(data) < 30:
            continue

        t1 = time.time()
        for agent in pop:
            agent.invest(data, states)
        print(time.time() - t1)

        scores = [agent.score for agent in pop]
        sor = np.argsort(scores)
        idx = sor[::-1]
        print(pop[sor[0]].score, ' <> ', pop[sor[len(pop) - 1]].score)
        best = pop[idx[0]]
        score = pop[idx[0]].score
        if len(last_5) == 5:
            last_5 = last_5[1:5]

        last_5.append(score)

        if len(last_5) == 5 and np.sum(np.array(last_5) - last_5[0]) == 0.:
            print('Early stopping criterion reached. Stopping....')
            last_5 = []
            break

        if iter >= MAX_ITER:
            break

        newpop = []
        for index in idx[:COPY_SEL]:
            newpop.append(CryptoRandomAgent(ticket, r_actions=pop[index].r_actions))

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

            newpop.extend([CryptoRandomAgent(ticket, r_actions=a),
                           CryptoRandomAgent(ticket, r_actions=b)])
            add += 2
        pop = newpop
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

print('data generation finished....')
