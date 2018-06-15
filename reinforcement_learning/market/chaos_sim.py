import numpy as np


def yorke(x, r):
    return x * r * (1. - x)


class CahosAgent:
    def __init__(self, ticket, invested=100000., r=None):
        self.score = 0
        self.ror_history = []
        self.dist = np.array([1.])
        self.ticket = ticket
        self.tr_cost = 2.
        if r is None:
            self.r = np.random.uniform(2.9, 3.9, 3)
        else:
            self.r = r
        self.invested = invested
        self.cash = invested
        self.shares = np.zeros(1)
        self.history = []
        self.actions = []
        self.x0_history = []
        self.x1_history = []
        self.x2_history = []
        self.state = []

    def invest(self, data):

        if len(data.keys()) == 0:
            return

        x = np.array([0.01, 0.01, 0.01])
        self.x0_history.append(x[0])
        self.x1_history.append(x[1])
        self.x2_history.append(x[2])

        for _ in range(20):
            x = yorke(x, self.r)
            self.x0_history.append(x[0])
            self.x1_history.append(x[1])
            self.x2_history.append(x[2])

        pct = data.pct_change().as_matrix()
        bench = data.pct_change().cumsum().as_matrix()

        for i in range(len(data)):
            prices = data.iloc[i].values
            portfolio = self.cash + np.dot(prices, self.shares)
            try:
                if np.isnan(portfolio):
                    portfolio = 0.
            except:
                print('portfolio:', portfolio)

            self.history.append(portfolio)

            ror = (portfolio - self.invested) / self.invested

            self.score_based_on_ror(ror)
            # self.score_based_on_beating_benchmark(ror, bench[i])

            self.ror_history.append(ror)

            x = yorke(x, self.r)
            self.x0_history.append(x[0])
            self.x1_history.append(x[1])
            self.x2_history.append(x[2])

            self.state.append(np.array([pct[i], bench[i]]))
            if x[2] >= 0.9:
                self.actions.append('H')
                continue
            elif x[0] >= 0.9 and sum(self.shares > 0) > 0:
                self.actions.append('S')
                self.cash = np.dot(self.shares, prices) - sum(self.shares > 0) * self.tr_cost
                self.shares = np.zeros(1)
            elif x[1] >= 0.9 and self.cash > prices:
                self.actions.append('B')
                portfolio = self.cash + np.dot(prices, self.shares)
                c = np.multiply(self.dist, portfolio)
                c = np.subtract(c, self.tr_cost)
                s = np.divide(c, prices)
                s = np.floor(s)
                self.shares = s
                self.cash = portfolio - np.dot(self.shares, prices) - len(s) * self.tr_cost
            else:
                self.actions.append('H')

    def score_based_on_ror(self, ror):
        if len(self.actions) > 0:
            if ror > 0.:
                self.score += 1
            elif ror < 0.:
                self.score -= 1

    def score_based_on_beating_benchmark(self, ror, benchmark, leverage=1.0):
        if len(self.actions) > 0:
            if ror*leverage >= benchmark:
                self.score += 1
            elif ror*leverage < benchmark:
                self.score -= 1
