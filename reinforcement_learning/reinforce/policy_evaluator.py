
import numpy as np
import time


class Agent:
    def __init__(self, r_actions):
        self.r_actions = r_actions
        self.ror_history = []
        self.dist = np.array([1.])
        self.tr_cost = 0.003
        self.invested = 100000.
        self.cash = 100000.
        self.shares = np.zeros(1)
        self.actions = []
        self.state = []
        self.coef = 1.
        self.gamma = 0.3

    def run(self, data, states):
        window = states.window
        data_window = len(data) - window
        self.history = np.zeros(data_window)
        self.ror_history = np.zeros(data_window)
        self.rewards = np.zeros_like(self.ror_history)

        for i in range(states.window, len(data)):
            self.one_action_step(i, states.window, data)

        self.ror_history = (self.history - self.invested) / self.invested
        if np.std(self.ror_history) != 0:
            self.rewards = (self.ror_history - states.bench[window:]) / np.std(self.ror_history)
        else:
            self.rewards = np.zeros_like(self.ror_history)

        self.disco_rewards = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            self.disco_rewards[t] = running_add

    def one_action_step(self, i, window, data):
        action = self.r_actions[i - window]
        prices = data[i]

        if i == 0:
            portfolio = self.cash + np.dot(prices, self.shares)
        else:
            portfolio = self.history[i - 1 - window]

        self.do_action(action, portfolio, prices)
        portfolio = self.cash + np.dot(prices, self.shares)
        self.history[i - window] = portfolio

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
