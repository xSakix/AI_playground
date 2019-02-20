import os
from datetime import datetime

from reinforcement_learning.crypto_market import crypto_sklearnclass_agent, crypto_bayes_agent
from reinforcement_learning.crypto_market.crypto_random_agent import CryptoRandomAgent
import sys

from reinforcement_learning.crypto_market.util import State

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import numpy as np
import matplotlib.pyplot as plt

from choose_best_score_agent import find_best_models
import pandas as pd

import time

np.warnings.filterwarnings('ignore')


def invest(args):
    args[0].invest(args[1], args[2])


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


def invest(arg):
    agent, data, state = arg
    agent.invest(data, state)


def invest2(agent, data, state):
    agent.invest(data, state)


def run_ga(clf_agent):
    MAX_POP = 100
    COPY_SEL = 2
    SELECTION = 10
    MUTATION_RATE = 0.7
    pop = init_pop(MAX_POP, clf_agent, data, states)
    last_best = None
    runs_score = []
    ga_score = []
    RUNS_SCORE_MAX = 5
    iter = 0

    while True:
        # t1 = time.time()
        for agent in pop:
            agent.run()
        # print('time:', time.time() - t1)

        sor = np.argsort([agent.score for agent in pop])
        idx = sor[::-1]
        best = pop[idx[0]]
        worst = pop[idx[-1]]
        score = pop[idx[0]].score
        ga_score.append(score)
        if len(runs_score) == RUNS_SCORE_MAX:
            runs_score = runs_score[1:RUNS_SCORE_MAX]

        runs_score.append(score)

        if len(runs_score) == RUNS_SCORE_MAX and np.sum(np.array(runs_score) - runs_score[0]) == 0.:
            print('Early stopping criterion reached. Stopping....')
            break

        new_pop = [CryptoRandomAgent(ticket, data, states, r_actions=pop[index].r_actions, agent=clf_agent) for index in
                   idx[:COPY_SEL]]

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

            new_pop.extend([CryptoRandomAgent(ticket, data, states, r_actions=a, agent=clf_agent),
                            CryptoRandomAgent(ticket, data, states, r_actions=b, agent=clf_agent)])
            add += 2
        pop = new_pop
        print('-'*80)
        print(iter, 'BEST->', score, ' | ', best.ror_history[-1], '|', np.mean(best.ror_history), '|', np.median(best.ror_history))
        print(iter, 'WORST->', worst.score, ' | ', worst.ror_history[-1], '|', np.mean(worst.ror_history), '|', np.median(worst.ror_history))
        iter += 1
        last_best = best

    print('\n best score:', best.score)
    print('ror:', best.ror_history[-1])
    x = best.state
    y = best.r_actions
    return x, y, best.ror_history[-1]


def init_pop(MAX_POP, clf_agent, data, states):
    return [
        CryptoRandomAgent(ticket, data, states, agent=clf_agent,
                          r_actions=np.random.randint(0, 3, size=len(data) - states.window))
        for _ in range(MAX_POP)]


if __name__ == '__main__':
    start_date = '2017-10-01'
    end_date = '2018-05-01'

    ticket = 'BTC-EUR'
    window = 30

    data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)
    data = data[ticket]
    print(start_date, ' - ', end_date)
    print(data.head(2))
    print(data.tail(2))
    print(len(data))

    states = State(window, data)

    it = 0
    dir_data = 'data_ga_periodic'
    dir_models = 'models_ga_periodic'

    classifier_scores = []
    classifier_rors = []
    classifier_acc_scores = []

    x_file = dir_data + '/x.npy'
    y_file = dir_data + '/y.npy'
    x_file_back = dir_data + '/x_back.npy'
    y_file_back = dir_data + '/y_back.npy'

    while it < 10:
        print('Iteration:', it)

        try:
            print(x_file, ' removing...')
            os.remove(x_file)
            print(y_file, ' removing...')
            os.remove(y_file)
        except:
            print('nothing to remove')

        if len(os.listdir(dir_models)) == 0:
            clf_agent = None
        else:
            _, clf_agent, _ = find_best_models(start_date=str(start_date), end_date=str(end_date),
                                               dir_models=dir_models + '/')
            classifier_scores.append(clf_agent.score)
            classifier_rors.append(clf_agent.ror_history[-1])
            print('Best score agent(%s - %s) is %s' % (start_date, end_date, clf_agent.model))

        x, y, best_ror = run_ga(clf_agent)
        if clf_agent is not None and clf_agent.ror_history[-1] > best_ror:
            print(clf_agent.ror_history[-1], ' > ', best_ror)
            continue

        if os.path.isfile(x_file):
            xx = np.load(x_file)
            x = np.concatenate([x, xx])

        if os.path.isfile(y_file):
            yy = np.load(y_file)
            y = np.concatenate([y, yy])

        np.save(x_file, x)
        np.save(y_file, y)
        np.save(x_file_back, x)
        np.save(y_file_back, y)
        it += 1

        for file in os.listdir(dir_models):
            os.remove(dir_models + '/' + file)
        acc_score = crypto_sklearnclass_agent.run_learning(dir_data + '/', dir_models + '/')
        classifier_acc_scores.append(acc_score)

    # evaluate results
    _, clf_agent, _ = find_best_models(start_date=str(start_date), end_date=str(end_date),
                                       dir_models=dir_models + '/')
    classifier_scores.append(clf_agent.score)
    classifier_rors.append(clf_agent.ror_history[-1])
    print('Best score agent(%s - %s) is %s' % (start_date, end_date, clf_agent.model))

    plt.plot(classifier_scores[1:])
    plt.show()

    plt.plot(classifier_rors)
    plt.show()

    plt.plot(classifier_acc_scores)
    plt.show()
