import os
from datetime import datetime

import sys

from reinforcement_learning.crypto_market.util import State
from reinforcement_learning.crypto_market_regressor.crypto_trader_regression_agent import CryptoRegressionAgent

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import numpy as np
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')


def find_best_models(start_date, end_date, dir_models='models_btc_eur/', ticket='BTC-EUR', debug=False):
    data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)
    data = data[data['date'] > str(start_date)]
    data = data[data['date'] < str(end_date)]

    print(start_date, " - ", end_date, " ,len = ", len(data))
    data = data[data[ticket] > 0.]
    data = data.reindex(method='bfill')

    print(data[[ticket]].head(2))
    print(data[[ticket]].tail(2))

    models = [dir_models + f for f in sorted(os.listdir(dir_models))]
    print(models)

    max = -99999999.
    max_ror = -99999999.
    max_agent = None
    best_ror_agent = None

    for model in models:
        agent = CryptoRegressionAgent(ticket, dir_models=model, coef=1.)
        agent.invest(data[ticket], window=30, debug=debug)
        print('testing:', model, ' => score:', agent.score, '=> ror:', agent.ror_history[-1], ' mean ror => ',
              np.mean(agent.ror_history))
        if max < agent.score:
            max = agent.score
            max_agent = agent
        if max_ror < agent.ror_history[-1]:
            max_ror = agent.ror_history[-1]
            best_ror_agent = agent

    # print(max_agent.clf.feature_importances_)

    return data, max_agent, best_ror_agent


def score_models(data, dir_models='models_btc_eur/', ticket='BTC-EUR'):
    result_score = {}
    result_ror = {}

    data = data[data[ticket] > 0.]
    data = data.reindex(method='bfill')

    print(data[[ticket]].head(2))
    print(data[[ticket]].tail(2))

    models = sorted(os.listdir(dir_models))

    for model in models:
        agent = CryptoRegressionAgent(ticket, dir_models=dir_models + str(model), coef=1.)
        agent.invest(data[ticket], window=30)
        print('testing:', agent.model, ' => score:', agent.score)
        result_score[agent.model] = agent.score
        result_ror[agent.model] = agent.ror_history[-1]

    return result_score, result_ror


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


def rank_by_score(dir_models):
    ranked_by_score = {}
    ranked_by_ror = {}
    ticket = 'BTC-EUR'
    # best_dir = 'best_models_btc_eur'
    #
    # if not os.path.isdir(best_dir):
    #     os.makedirs(best_dir)
    #
    # best_models_dir = os.listdir(best_dir)

    start_date = '2011-08-07'
    end_date = '2018-06-27'
    df_adj_close = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)

    for it in range(100):

        print('-' * 80)
        print(it)
        print('-' * 80)

        data, start, end = get_data_random_dates(df_adj_close, 2011, 2018)
        while len(data) < 30:
            data, start, end = get_data_random_dates(df_adj_close, 2011, 2018)

        print(start, ' - ', end)

        result_score, result_ror = score_models(data, ticket=ticket, dir_models=dir_models)
        for key in result_score.keys():
            if key in ranked_by_score.keys():
                ranked_by_score[key].append(result_score[key])
                ranked_by_ror[key].append(result_ror[key])
            else:
                ranked_by_score[key] = [result_score[key]]
                ranked_by_ror[key] = [result_ror[key]]

    import pandas as pd
    df = pd.DataFrame(columns=['model', 'score', 'ror'])
    for key in ranked_by_score.keys():
        median_score = np.mean(np.array(ranked_by_score[key]))
        median_ror = np.mean(np.array(ranked_by_ror[key]))
        df = df.append({'model': key, 'score': median_score, 'ror': median_ror}, ignore_index=True)

    best_by_score = df.sort_values(by=['score'], ascending=False).head(1)
    print(best_by_score)
    best_by_ror = df.sort_values(by=['ror'], ascending=False).head(1)
    print(best_by_ror)

    result_list = best_by_score['model'].tolist()
    result_list.extend(best_by_ror['model'].tolist())

    # from shutil import copyfile
    # for m in result_list:
    #     split = m.split('/')
    #     _dir = split[0]
    #     file = split[1]
    #     if file not in best_models_dir:
    #         copyfile(m, best_dir + '/' + _dir + file)


def best_of_best(model_dir='best_models_btc_eur/', debug=False):
    start_date = '2017-04-01'
    end_date = '2018-04-01'
    # start_date = '2018-07-01'
    # end_date = '2018-09-14'

    ticket = 'BTC-EUR'
    best_dir = model_dir

    data, max_agent, best_ror_agent = find_best_models(start_date, end_date, ticket=ticket, dir_models=best_dir,
                                                       debug=debug)
    data = data[ticket].reset_index(drop=True).fillna(method="bfill")
    print(data.head(2))
    print(data.tail(2))

    window = 30
    states = State(window, data)
    print(states.bench[0])
    print(states.bench[1])
    print(states.bench[-2])
    print(states.bench[-1])
    print(len(data) - window)

    plt.plot(states.bench[window:], color='black')
    plt.plot(max_agent.ror_history, color='red')
    plt.plot(best_ror_agent.ror_history, color='blue')
    plt.legend(['bench', max_agent.model, best_ror_agent.model])
    plt.show()

    print('best(score):', max_agent.model)
    print('ror:', max_agent.ror_history[-1])
    print('portfolio:', max_agent.history[-1])

    print('best(ror):', best_ror_agent.model)
    print('ror:', best_ror_agent.ror_history[-1])
    print('portfolio:', best_ror_agent.history[-1])


if __name__ == "__main__":
    # rank_by_score('/home/martin/model/')
    best_of_best(model_dir='/home/martin/model/', debug=True)
    # beast_of_best()
    # start_date = '2018-05-01'
    # end_date = '2018-08-22'
    # ticket = 'BTC-EUR'
    # data = load_all_data_from_file2('btc_etf_data_adj_close_1.csv', start_date, end_date)
    # data = data[data['date'] > str(start_date)]
    # data = data[data['date'] < str(end_date)]
    #
    # print(start_date, " - ", end_date, " ,len = ", len(data))
    # data = data[data[ticket] > 0.]
    # data = data.reindex(method='bfill')
    #
    # print(data[ticket].head(2))
    # print(data[ticket].tail(2))
    # bench = data[ticket].apply(lambda x: (x / data[ticket].iloc[0]) - 1)
    # print(bench)
