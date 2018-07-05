import pandas as pd
from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent

import numpy as np
from sklearn.preprocessing import StandardScaler


class CryptoRandomAgent:
    def __init__(self, ticket, use_trader=False, invested=100000., model=None, agent=None, r_actions=np.empty(0)):
        self.r_actions = r_actions
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
        self.model = model
        self.agent = agent

    def invest(self,
               data,
               window,
               pct,
               bench,
               mean,
               median,
               lowerbb,
               upperbb):

        if len(data.keys()) == 0:
            return

        if len(self.r_actions) == 0:
            self.r_actions = np.random.randint(0, 3, size=len(data) - window)

        self.ror_history = np.empty(len(pct))
        self.ror_history[:] = np.nan

        for i in range(window, len(data)):

            action = self.get_action(i - window)
            prices = data.iloc[i].values
            portfolio = self.cash + np.dot(prices, self.shares)

            try:
                if np.isnan(portfolio):
                    portfolio = 0.
            except:
                print('portfolio:', portfolio)

            self.history.append(portfolio)

            self.ror_history[i] = (portfolio - self.invested) / self.invested
            if self.use_trader and i > window:
                # print('%.3f : %.3f : %.3f' % (ror, self.agent.ror_history[i], bench[i]))
                self.score_based_on_beating_trader(self.ror_history[i], self.agent.ror_history[i])
            elif i > window:
                # self.score_based_on_ror(ror)
                self.score_based_on_beating_benchmark(self.ror_history[i], bench[i])

            input = [
                self.ror_history[i],
                bench[i],
                pct[i],
                lowerbb[i],
                mean[i],
                median[i],
                upperbb[i]]

            self.state.append(np.array(input))

            self.do_action(action, portfolio, prices)

        df = pd.DataFrame(self.ror_history)
        df.fillna(method="bfill", inplace=True)
        self.ror_history = df.as_matrix()
        self.state = np.array(self.state)

    def do_action(self, action, portfolio, prices):
        if action == 1 and sum(self.shares > 0) > 0:
            to_sell = self.coef * self.shares
            sold = np.dot(to_sell, prices)
            self.cash += sold - sold * self.tr_cost
            self.shares = self.shares - to_sell
            # portfolio = self.cash + np.dot(prices, self.shares)
            # print('selling ', to_sell, ' portfolio=', portfolio, 'cash=', self.cash,'shares=',self.shares)
        elif action == 2 and (self.coef * self.cash - self.tr_cost * self.coef * self.cash) > 0.000000001 * prices:
            c = self.cash * self.coef
            cost = np.multiply(self.tr_cost, c)
            c = np.subtract(c, cost)
            s = np.divide(c, prices)
            self.shares += s
            self.cash = portfolio - np.dot(self.shares, prices) - cost
            # portfolio = self.cash + np.dot(prices, self.shares)
            # print('buying ', s, ' portfolio=', portfolio, 'cash=', self.cash,'shares=',self.shares)

    def get_action(self, i):
        try:
            action = self.r_actions[i]
        except:
            action = np.random.randint(0, 3)

        return action

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
        if ror >= ror_agent:
            self.score += 1
        elif ror < ror_agent:
            self.score -= 1
