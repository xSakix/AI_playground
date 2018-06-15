import pandas as pd
from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent

import numpy as np
from sklearn.preprocessing import StandardScaler


class CryptoRandomAgent:
    def __init__(self, ticket, use_trader=False, invested=100000.):
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
        self.coef = 0.5
        self.use_trader = use_trader

    def invest(self, data, window=30):

        if len(data.keys()) == 0:
            return

        data.fillna(method="bfill", inplace=True)

        if self.use_trader:
            agent = CryptoTraderAgent(self.ticket)
            agent.invest(data, window=30)

        pct = data.pct_change().as_matrix()
        bench = data.pct_change().cumsum().as_matrix()
        data_1 = pd.DataFrame(pct)
        mean = data_1.rolling(window=window).mean().as_matrix()
        median = data_1.rolling(window=window).median().as_matrix()
        std = data_1.rolling(window=window).std().as_matrix()
        upperbb = mean + (2 * std)
        lowerbb = mean - (2 * std)

        self.ror_history = np.empty(len(pct))
        self.ror_history[:] = np.nan

        for i in range(window, len(data)):

            action = np.random.randint(0, 3)
            prices = data.iloc[i].values
            portfolio = self.cash + np.dot(prices, self.shares)

            try:
                if np.isnan(portfolio):
                    portfolio = 0.
            except:
                print('portfolio:', portfolio)

            self.history.append(portfolio)

            ror = (portfolio - self.invested) / self.invested

            if self.use_trader:
                self.score_based_on_beating_trader(ror, agent.ror_history[i])
            else:
                # self.score_based_on_ror(ror)
                self.score_based_on_beating_benchmark(ror, bench[i])

            self.ror_history[i] = ror

            self.r_actions.append(action)
            input = [
                self.ror_history[i],
                bench[i],
                pct[i],
                lowerbb[i],
                mean[i],
                median[i],
                upperbb[i]]
            self.state.append(np.array(input))

            if action == 0:
                self.actions.append('H')
                continue
            elif action == 1 and sum(self.shares > 0) > 0:
                self.actions.append('S')
                to_sell = self.coef * self.shares
                sold = np.dot(to_sell, prices)
                self.cash += sold - sold * self.tr_cost
                self.shares = self.shares - to_sell
                # portfolio = self.cash + np.dot(prices, self.shares)
                # print('selling ', to_sell, ' portfolio=', portfolio, 'cash=', self.cash,'shares=',self.shares)
            elif action == 2 and (self.coef * self.cash - self.tr_cost * self.coef * self.cash) > 0.000000001 * prices:
                self.actions.append('B')
                c = self.cash * self.coef
                cost = np.multiply(self.tr_cost, c)
                c = np.subtract(c, cost)
                s = np.divide(c, prices)
                self.shares += s
                self.cash = portfolio - np.dot(self.shares, prices) - cost
                # portfolio = self.cash + np.dot(prices, self.shares)
                # print('buying ', s, ' portfolio=', portfolio, 'cash=', self.cash,'shares=',self.shares)
            else:
                self.actions.append('H')

        df = pd.DataFrame(self.ror_history)
        df.fillna(method="bfill", inplace=True)
        self.ror_history = df.as_matrix()
        self.state = np.array(self.state)
        self.r_actions = np.array(self.r_actions)

    def score_based_on_ror(self, ror):
        if len(self.actions) > 0:
            if ror > 0.:
                self.score += 1
            elif ror < 0.:
                self.score -= 1

    def score_based_on_beating_benchmark(self, ror, benchmark, leverage=1.0):
        if len(self.actions) > 0:
            if ror * leverage >= benchmark:
                self.score += 1
            elif ror * leverage < benchmark:
                self.score -= 1

    def score_based_on_beating_trader(self, ror, ror_agent):
        if len(self.actions) > 0:
            if ror >= ror_agent:
                self.score += 1
            elif ror < ror_agent:
                self.score -= 1
