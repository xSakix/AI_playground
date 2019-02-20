import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

from reinforcement_learning.crypto_market.util import State, my_score

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

np.warnings.filterwarnings('ignore')


class CryptoRegressionAgent:
    def __init__(self, ticket, dir_models, invested=100000., coef=.5):
        self.r_actions = []
        self.score = 0
        self.ror_history = []
        self.dist = np.array([1.])
        self.ticket = ticket
        self.tr_cost = 0.003
        self.invested = invested
        self.cash = invested
        self.shares = np.zeros(1)
        self.history = []
        self.actions = []
        self.state = []
        self.model = dir_models
        for model in os.listdir(dir_models):
            print(model)
            if 'buy' in str(model):
                self.clf_buy = joblib.load(dir_models + '/' + model)
            if 'sell' in str(model):
                self.clf_sell = joblib.load(dir_models + '/' + model)
            if 'hold' in str(model):
                self.clf_hold = joblib.load(dir_models + '/' + model)
        self.coef = coef

    def invest(self, data, window=30, debug=False):

        if len(data.keys()) == 0:
            return

        data.fillna(method="bfill", inplace=True)
        states = State(window, data)

        self.state = np.array([states.get_state(i) for i in range(window, len(data))])
        np.set_printoptions(suppress=True)

        self.ror_history = np.full(len(data) - window, 0.)
        self.history = np.full(len(data) - window, 0.)

        self.r_actions = np.empty((len(data) - window, 3))
        self.r_actions[:, 0] = self.clf_hold.predict(self.state[:, [0, 2, 3]])
        self.r_actions[:, 1] = self.clf_sell.predict(self.state[:, [0, 2, 3]])
        self.r_actions[:, 2] = self.clf_buy.predict(self.state[:, [0, 2, 3]])
        # [print(a) for a in self.r_actions]
        self.r_actions = np.argmax(self.r_actions, axis=1)

        for i in range(window, len(data)):
            # portfolio = self.cash + data.iloc[i] * self.shares
            # print(portfolio, '|', self.cash, '|', self.shares)
            self.one_step(data.iloc[i], debug, i, window)
            self.ror_history[i - window] = self.history[i - window] / self.invested - 1.

        # self.ror_history = (self.history - self.invested) / self.invested
        # [print(r) for r in self.ror_history]
        self.score = my_score(self.ror_history, states.bench)

    def one_step(self, prices, debug, i, window):

        portfolio = self.cash + prices * self.shares
        self.history[i - window] = portfolio
        # print(portfolio)
        self.do_action(self.r_actions[i - window], portfolio, prices, debug)

    def do_action(self, action, portfolio, prices, debug):
        SELL = 1
        BUY = 2
        # reverted
        # SELL = 2
        # BUY = 1
        if action == SELL:
            return self.sell(prices, debug)
        elif action == BUY:
            return self.buy(portfolio, prices, debug)
        else:
            return 0

    def buy(self, portfolio, prices, debug):
        available_cash = self.cash * self.coef
        cost = self.tr_cost * available_cash
        available_cash = available_cash - cost

        if available_cash < 0.001 * prices:
            return 0

        if debug:
            print('B[%f]' % prices)

        s = available_cash / prices
        self.shares += s
        self.cash = portfolio - self.shares * prices - cost

        return 2

    def sell(self, prices, debug):
        to_sell = self.coef * self.shares

        if to_sell == 0:
            return 0

        if debug:
            print('S[%f]' % prices)

        sold = to_sell * prices
        self.cash += sold - sold * self.tr_cost
        self.shares = self.shares - to_sell

        return 1


if __name__ == '__main__':
    # start_date = '2018-04-01'
    # end_date = '2018-09-14'
    # start_date = '2017-04-01'
    # end_date = '2018-04-01'
    ticket = 'BTC-EUR'
    # model = 'models_ga_periodic/BTC_EUR_random_forrest_0.pkl'
    # model = '/home/martin/model/BTC_EUR_mlp_5.pkl'
    # model = '/home/martin/model/BTC_EUR_random_forrest_4.pkl'
    # model = '/home/martin/models_eu/BTC_EUR_random_forrest_49.pkl'
    dir_models = '/home/martin/model/5'
    df_adj_close = pd.read_csv('/home/martin/data/coinbaseEUR.csv')
    data = df_adj_close[ticket]

    # data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)

    agent = CryptoRegressionAgent(ticket, dir_models=dir_models, invested=100., coef=.5)
    window = 30
    # agent.invest(data[ticket], window=window, debug=True)
    agent.invest(data, window=window, debug=True)
    print('testing:', dir_models, ' => score:', agent.score, '=> ror:', agent.ror_history[-1], ' mean ror => ',
          np.mean(agent.ror_history))

    ror_bah = data.apply(lambda x: (x / data[ticket].iloc[0]) - 1.).as_matrix()[window:]
    plt.plot(ror_bah, color='black')
    plt.plot(agent.ror_history, color='red')
    plt.legend(['bench', dir_models], loc='upper left')
    plt.show()

    print('best(score):', dir_models)
    print('ror:', agent.ror_history[-1])
    print('portfolio:', agent.history[-1])
