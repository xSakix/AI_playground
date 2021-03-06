import multiprocessing
import threading

import numpy as np
from reinforcement_learning.crypto_market.util import my_score
import time


class CryptoRandomAgent:
    def __init__(self, ticket, data, states, invested=100000., agent=None, r_actions=np.empty(0), coef=0.5):
        self.r_actions = r_actions
        self.score = 0
        self.ror_history = []
        self.dist = np.array([1.])
        self.ticket = ticket
        self.tr_cost = 0.003
        self.invested = invested
        self.cash = invested
        self.shares = np.zeros(1)
        self.actions = []
        self.state = []
        self.coef = coef
        self.agent = agent
        self.data = data
        self.states = states
        self.history = np.full(len(self.data) - self.states.window, fill_value=0.)
        self.ror_history = np.full(len(self.data) - self.states.window, fill_value=0.)

    def run(self):
        for i in range(self.states.window, len(self.data)):
            self.one_action_step(i)

        self.ror_history = (self.history - self.invested) / self.invested
        self.ror_history = np.nan_to_num(self.ror_history)

        try:
            ror_agent = np.nan_to_num(np.copy(self.agent.ror_history))
            self.score = my_score(self.ror_history, ror_agent)
        except:
            self.score = my_score(self.ror_history, self.states.bench)

        self.state = np.array(self.state)

    def one_action_step(self, i):
        action = self.r_actions[i - self.states.window]
        prices = self.data.iloc[i]
        portfolio = self.cash + np.dot(prices, self.shares)
        self.history[i - self.states.window] = portfolio
        self.state.append(self.states.get_state(i))
        action = self.do_action(action, portfolio, prices)
        self.r_actions[i - self.states.window] = action

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

    def score_based_on_ror(self, ror):
        if ror > 0.:
            self.score += 1

    def score_based_on_beating_benchmark(self, ror, benchmark):
        if ror > 0. and ror >= benchmark:
            self.score += 1 + int(ror / benchmark)

    def score_based_on_beating_trader(self, ror, ror_agent):
        if ror > 0. and ror >= ror_agent:
            try:
                self.score += 1 + int(ror / ror_agent)
            except:
                self.score += 1


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
