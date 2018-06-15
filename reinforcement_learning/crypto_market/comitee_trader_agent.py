import pandas as pd

import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from reinforcement_learning.crypto_market.crypto_trader_agent import CryptoTraderAgent


class ComiteeTraderAgent:
    def __init__(self, ticket, invested=100000.):
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

    def invest(self, data, window=30):

        if len(data.keys()) == 0:
            return

        import os
        models = sorted(os.listdir('models'))

        self.agents = []
        for model in models:
            agent = CryptoTraderAgent(self.ticket, model='models/' + str(model))
            agent.invest(data, window=30)
            self.agents.append(agent)

        data.fillna(method="bfill", inplace=True)

        pct = data.pct_change().as_matrix()
        bench = data.pct_change().cumsum().as_matrix()
        data_1 = pd.DataFrame(pct)
        mean = data_1.rolling(window=window).mean().as_matrix()
        median = data_1.rolling(window=window).median().as_matrix()
        std = data_1.rolling(window=window).std().as_matrix()
        upperbb = mean + (2 * std)
        lowerbb = mean - (2 * std)

        self.ror_history = np.empty(len(data))
        self.ror_history[:] = np.nan

        for i in range(window + 1, len(data)):

            prices = data.iloc[i].values

            portfolio = self.cash + np.dot(prices, self.shares)

            try:
                if np.isnan(portfolio):
                    portfolio = 0.
            except:
                print('portfolio:', portfolio)

            self.history.append(portfolio)

            ror = (portfolio - self.invested) / self.invested

            self.ror_history[i] = ror

            input = [[self.ror_history[i],
                      bench[i],
                      pct[i - 1],
                      lowerbb[i - 1],
                      mean[i - 1],
                      median[i - 1],
                      upperbb[i - 1]]]
            input = np.array(input)

            action = self.get_action(i)
            self.state.append(input)
            self.r_actions.append(action)

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

    def get_action(self, index):
        actions = np.array([agent.r_actions[index-31] for agent in self.agents])
        actions, counts = np.unique(actions, return_counts=True)
        return actions[np.argmax(counts)]
