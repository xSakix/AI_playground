import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import sys

from reinforcement_learning.crypto_market.util import State, my_score

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

np.warnings.filterwarnings('ignore')


class CryptoTraderAgent:
    def __init__(self, ticket, model='models/decision_tree.pkl', invested=100000., coef=.5):
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
        self.model = model
        self.clf = joblib.load(model)
        self.coef = coef
        self.last_buy_price = None
        self.last_sell_price = None
        self.action_map = {1: 1, 2: 2}

    def reverse_action_map(self):
        if self.action_map[1] == 1:
            self.action_map = {1: 2, 2: 1}
        else:
            self.action_map = {1: 1, 2: 2}

    def invest(self, data, window=30, debug=False):

        if len(data.keys()) == 0:
            return

        data.fillna(method="bfill", inplace=True)

        states = State(window, data)
        # print(data.head(2))
        # print(data.tail(2))
        # print(states.bench[0])
        # print(states.bench[1])
        # print(states.bench[-2])
        # print(states.bench[-1])

        self.state = np.array([states.get_state(i) for i in range(window, len(data))])
        np.set_printoptions(suppress=True)
        # print(self.state)

        self.ror_history = np.full(len(data) - window, 0.)
        self.history = np.full(len(data) - window, 0.)
        self.r_actions = self.clf.predict(self.state)
        # print(self.r_actions)

        for i in range(window, len(data)):
            portfolio = self.cash + data.iloc[i] * self.shares
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
        self.decide_to_reverse_actions(self.r_actions[i - window],prices)

    def decide_to_reverse_actions(self, action, prices):
        SELL = 1
        BUY = 2

        if action == self.action_map[SELL]:
            if self.last_buy_price is not None and self.last_buy_price > prices:
                self.reverse_action_map()
                print('reverting')
        elif action == self.action_map[BUY]:
            if self.last_sell_price is not None and self.last_sell_price < prices:
                self.reverse_action_map()
                print('reverting')

    def do_action(self, action, portfolio, prices, debug):
        SELL = 1
        BUY = 2

        if action == self.action_map[SELL]:
            return self.sell(prices, debug)
        elif action == self.action_map[BUY]:
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

        self.last_buy_price = prices

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

        self.last_sell_price = prices

        return 1


if __name__ == '__main__':
    start_date = '2018-07-01'
    end_date = '2018-09-14'
    ticket = 'BTC-EUR'
    # model = 'models_ga_periodic/BTC_EUR_random_forrest_0.pkl'
    # model = '/home/martin/model/BTC_EUR_mlp_5.pkl'
    # model = '/home/martin/Projects/AI_playground/reinforcement_learning/crypto_market/learned_classifiers/20180916/BTC_EUR_random_forrest_4.pkl'
    model = '/home/martin/models_eu/BTC_EUR_random_forrest_558.pkl'
    data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)

    agent = CryptoTraderAgent(ticket, model=model, invested=100., coef=.5)
    window = 30
    agent.invest(data[ticket], window=window, debug=True)
    print('testing:', model, ' => score:', agent.score, '=> ror:', agent.ror_history[-1], ' mean ror => ',
          np.mean(agent.ror_history))

    ror_bah = data[ticket].apply(lambda x: (x / data[ticket].iloc[0]) - 1.).as_matrix()[window:]
    plt.plot(ror_bah, color='black')
    plt.plot(agent.ror_history, color='red')
    plt.legend(['bench', agent.model], loc='upper left')
    plt.show()

    print('best(score):', agent.model)
    print('ror:', agent.ror_history[-1])
    print('portfolio:', agent.history[-1])
