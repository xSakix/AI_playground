import multiprocessing
import threading

import numpy as np
from reinforcement_learning.crypto_market.util import my_score
import time


class RecordingAgent:
    def __init__(self, data, states):
        self.data = data
        self.states = states
        self.history = np.full(len(data) - states.window, fill_value=0.)
        self.r_actions = np.zeros(len(data) - states.window)
        self.ror_history = []
        self.dist = np.array([1.])
        self.tr_cost = 0.003
        self.invested = 100000.
        self.cash = 100000.
        self.shares = np.zeros(1)
        self.actions = []
        self.state = []
        self.coef = 1.
        self.ror_history = np.full(len(data) - states.window, fill_value=0.)
        self.rewards = np.zeros_like(self.ror_history)
        self.discount_rate = 0.99
        self.q = np.zeros_like(self.ror_history)

    def run(self, action, t):
        window = self.states.window

        index = t - window

        self.r_actions[index] = action
        self.one_action_step(t, window, self.data)

        self.ror_history[index] = (self.history[index] - self.invested) / self.invested
        # if self.ror_history[index] > 0.:
        #     self.rewards[index] = 1

        if np.std(self.ror_history[:index]) != 0 and len(self.ror_history[self.ror_history > 0]) > 0:
            self.rewards[index] = (self.ror_history[index] - self.states.bench[t]) / np.std(self.ror_history[:index])
        else:
            self.rewards[index] = 0

        # if index > 0:
        #     if self.history[index -1] == 0:
        #         self.rewards[index] = 0
        #     else:
        #         self.rewards[index] = self.history[index]/self.history[index-1] -1.

        self.disco_rewards = np.zeros(index + 1)
        running_add = 0
        for tt in reversed(range(index+1)):
            running_add = running_add * self.discount_rate + self.rewards[tt]
            self.disco_rewards[tt] = running_add

        # print(self.disco_rewards)

        if self.disco_rewards.std() != 0:
            self.disco_rewards -= self.disco_rewards.mean()/self.disco_rewards.std()

        self.disco_rewards = self.disco_rewards[index:index+1]


    def one_action_step(self, t, window, data):
        action = self.r_actions[t - window]
        prices = data[t]
        portfolio = self.cash + np.dot(prices, self.shares)
        self.do_action(action, portfolio, prices)
        self.history[t - window] = self.cash + np.dot(prices, self.shares)

    def do_action(self, action, portfolio, prices):
        if action == 1:
            return self.sell(prices)
        elif action == 2:
            return self.buy(portfolio, prices)
        else:
            return 0

    def buy(self, portfolio, prices):
        available_cash = self.cash * self.coef
        cost = np.multiply(self.tr_cost, available_cash)
        available_cash = np.subtract(available_cash, cost)

        if available_cash < 0.001 * prices:
            return 0

        s = np.divide(available_cash, prices)
        self.shares += s
        self.cash = portfolio - np.dot(self.shares, prices) - cost

        return 2

    def sell(self, prices):
        to_sell = self.coef * self.shares
        if to_sell == 0:
            return 0

        sold = np.dot(to_sell, prices)
        self.cash += sold - sold * self.tr_cost
        self.shares = self.shares - to_sell

        return 1


if __name__ == "__main__":
    from reinforcement_learning.crypto_market.util import State

    import sys

    sys.path.insert(0, '../../../etf_data')
    from etf_data_loader import load_all_data_from_file2

    start_date = '2017-05-01'
    end_date = '2018-05-01'

    ticket = 'BTC-EUR'
    window = 30
    data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)
    print(start_date, ' - ', end_date)
    data = data[ticket].reset_index(drop=True).fillna(method="bfill")
    print(data.head(2))
    print(data.tail(2))
    print(len(data))

    states = State(window, data)

    t1 = time.time()
    for _ in range(100):
        agent = CryptoRandomAgent(ticket, data, states,
                                  r_actions=np.random.randint(0, 3, size=len(data) - states.window),
                                  coef=0.5)
        agent.run()
    print('time:', time.time() - t1)

    print(len(agent.state))
    print(len(agent.r_actions))
