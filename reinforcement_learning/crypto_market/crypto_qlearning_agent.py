# cython: language_level=3, boundscheck=False

import pandas as pd
import numpy as np
import sys
import os

from Cython.Shadow import array

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')

ROUNDING_FACTOR = 2
MAX_ITERATIONS = 10000


# can be changed to iterable
class State:
    def __init__(self, window, pct, bench, mean, median, lowerbb, upperbb):
        self.window = window
        self.pct = pct
        self.bench = bench
        self.mean = mean
        self.median = median
        self.lowerbb = lowerbb
        self.upperbb = upperbb

    def get_state(self, i):
        # return str([self.pct[i], self.lowerbb[i], self.mean[i], self.median[i], self.upperbb[i]])
        return str([self.pct[i], self.lowerbb[i], self.upperbb[i]])


class CryptoQLearnedAgentSimulator:
    def __init__(self, ticket, q_table):
        self.q_table = q_table
        self.dist = np.array([1.])
        self.ticket = ticket
        self.tr_cost = 0.003
        self.invested = 100000.
        self.coef = 1.
        self.cash = self.invested
        self.shares = np.zeros(1)
        self.history = []
        self.state = []
        self.ror_history = []

    def invest(self, data, states):

        if len(data.keys()) == 0:
            return
        self.ror_history = np.empty(len(states.pct))
        self.ror_history[:] = np.nan

        for i in range(states.window, len(data)):

            prices = data.iloc[i].values
            portfolio = np.nan_to_num(self.cash + np.dot(prices, self.shares))
            self.history.append(portfolio)
            self.ror_history[i] = (portfolio - self.invested) / self.invested

            if i == len(data) - 1:
                break

            self.state.append(np.array(states.get_state(i)))
            self.do_action(self.get_action(states.get_state(i)), portfolio, prices)

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
        elif action == 2 and (self.coef * self.cash - self.tr_cost * self.coef * self.cash) > 0.000000001 * prices:
            c = self.cash * self.coef
            cost = np.multiply(self.tr_cost, c)
            c = np.subtract(c, cost)
            s = np.divide(c, prices)
            self.shares += s
            self.cash = portfolio - np.dot(self.shares, prices) - cost

    def get_action(self, state):
        if state not in self.q_table.index:
            print('Missing state:', state)
            self.q_table = self.q_table.append(pd.Series([0, 0, 0], index=self.q_table.columns, name=state))

        return self.q_table.loc[state, :].values.argmax()


def end_condition(it, max_it, ror, actual_ror):
    if ror is None:
        return actual_ror > 0. and it > max_it
    else:
        return ror < actual_ror or it > 1000


def reward(ror, benchmark):
    if ror >= benchmark:
        return 1 + ror / benchmark
    return 0


class CryptoQLearningAgent:
    def __init__(self, ticket, invested=100000., q_table=None):
        self.lr = 0.1
        self.epsilon = 1.0
        self.gamma = 0.6

        if q_table is None:
            self.q_table = pd.DataFrame(columns=['0', '1', '2'])
        else:
            self.q_table = q_table

        self.dist = np.array([1.])
        self.ticket = ticket
        self.tr_cost = 0.003
        self.invested = invested
        self.coef = 1.
        self.cash = self.invested
        self.shares = np.zeros(1)
        self.history = []
        self.state = []
        self.ror_history = []
        self.best_table = None
        self.best_ror = None

    def _populate_q_table(self, states):
        for i in range(states.window, len(states.pct)):
            state = states.get_state(i)
            if state not in self.q_table.index:
                self.q_table = self.q_table.append(pd.Series([0, 0, 0], index=self.q_table.columns, name=state))

    def run_learning(self, data, states, last_ror=None):

        if len(self.q_table) == 0:
            print('populating table...')
            self._populate_q_table(states)
        else:
            print('using existing qtable...')

        rors = []

        it = 0
        max_it = MAX_ITERATIONS
        print('Starting learning...')
        while True:
            self.invest(data, states, it)
            if last_ror is not None:
                print('\r{} RoR of agent is: {}/{}'.format(it, self.ror_history[-1], last_ror), end='')
            else:
                print('\r{} RoR of agent is: {}/{}'.format(it, self.ror_history[-1], self.best_ror), end='')

            rors.append(self.ror_history[-1])
            if self.best_ror is None or self.best_ror < self.ror_history[-1]:
                self.best_ror = self.ror_history[-1]
                self.best_table = self.q_table.copy()
            it += 1
            if end_condition(it, max_it, last_ror, self.ror_history[-1]):
                if last_ror is not None:
                    print('\nStopping learning(had last ror)...%d/%d | %f/%f' % (
                        it, max_it, last_ror, self.ror_history[-1]))
                else:
                    print('\nStopping learning...%d/%d | %f/%f' % (it, max_it, self.best_ror, self.ror_history[-1]))
                print(self.best_ror, self.ror_history[-1])
                break
            self.epsilon = self.epsilon / max_it

        import datetime
        now = str(datetime.datetime.today()).replace('-', '').replace(' ', '_').replace(':', '_').replace('.', '_');

        self.best_table.to_csv('qtables_short/' + now + '_qtable.csv')

        # plt.plot(rors)
        # plt.show()

    def invest(self, data, states, it):

        if len(data.keys()) == 0:
            return

        if it > 0:
            self.epsilon = self.epsilon / it

        self.cash = self.invested
        self.shares = np.zeros(1)
        self.history = []
        self.state = []
        self.ror_history = np.empty(len(states.pct))
        self.ror_history[:] = np.nan

        for i in range(states.window, len(data)):

            prices = data.iloc[i].values
            portfolio = np.nan_to_num(self.cash + np.dot(prices, self.shares))
            self.history.append(portfolio)
            self.ror_history[i] = (portfolio - self.invested) / self.invested

            if i == len(data) - 1:
                break

            self.state.append(np.array(states.get_state(i)))
            action = self.get_action(states.get_state(i))
            # self.score_based_on_ror(ror)

            self.do_action(action, portfolio, prices)
            self.learn(states.get_state(i), str(action),
                       reward(self.ror_history[i], states.bench[i]),
                       states.get_state(i + 1))

        df = pd.DataFrame(self.ror_history)
        df.fillna(method="bfill", inplace=True)
        self.ror_history = df.as_matrix()
        self.state = np.array(self.state)

    def learn(self, state, action, _reward, new_state):
        q_predict = self.q_table.loc[state, action]
        q_target = _reward + self.gamma * self.q_table.loc[new_state, :].max()
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

    def do_action(self, action, portfolio, prices):
        if action == 1 and sum(self.shares > 0) > 0:
            to_sell = self.coef * self.shares
            sold = np.dot(to_sell, prices)
            self.cash += sold - sold * self.tr_cost
            self.shares = self.shares - to_sell
        elif action == 2 and (self.coef * self.cash - self.tr_cost * self.coef * self.cash) > 0.000000001 * prices:
            c = self.cash * self.coef
            cost = np.multiply(self.tr_cost, c)
            c = np.subtract(c, cost)
            s = np.divide(c, prices)
            self.shares += s
            self.cash = portfolio - np.dot(self.shares, prices) - cost

    def get_action(self, observation):
        if observation not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0, 0, 0], index=self.q_table.columns, name=observation))

        state_action = self.q_table.loc[observation, :]
        if np.random.rand() < self.epsilon and np.sum(state_action) > 0.:
            action = state_action.values.argmax()
        else:
            action = np.random.randint(0, 3)

        return action


def learn_agent(use_last_run=False):
    start_date = '2017-10-01'
    end_date = '2018-06-27'
    prefix = 'btc_'
    ticket = 'BTC-EUR'

    data = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)

    data = data[[ticket]]
    data = data.reset_index(drop=True)
    data.fillna(method="bfill", inplace=True)
    print(data.head(2))
    print(data.tail(2))

    states = load_states(data, 30)

    if use_last_run:
        max_qtable = None
        last_ror = None
        for table in os.listdir('q_tables'):
            q_table = pd.read_csv('q_tables/' + table)
            q_table = q_table.set_index('Unnamed: 0')
            agent = CryptoQLearnedAgentSimulator(ticket, q_table=q_table)
            agent.invest(data, states)

            if last_ror is None or last_ror < agent.ror_history[-1]:
                last_ror = agent.ror_history[-1]
                max_qtable = agent.q_table.copy()

    agent = CryptoQLearningAgent(ticket)
    if use_last_run:
        agent.run_learning(data, states=states, last_ror=last_ror)
    else:
        agent.run_learning(data, states=states)


def test_agent(show_graph=False):
    # start_date = '2011-08-07'
    start_date = '2017-10-01'
    end_date = '2018-06-27'
    prefix = 'btc_'
    ticket = 'BTC-EUR'

    data = load_all_data_from_file2(prefix + 'etf_data_adj_close.csv', start_date, end_date)

    data = data[[ticket]]
    data = data.reset_index(drop=True)
    data.fillna(method="bfill", inplace=True)
    print(data.head(2))
    print(data.tail(2))

    states = load_states(data, 30)

    legend = []

    result = {}

    for table in os.listdir('qtables_short'):
        q_table = pd.read_csv('qtables_short/' + table)
        q_table = q_table.set_index('Unnamed: 0')
        agent = CryptoQLearnedAgentSimulator(ticket, q_table)
        agent.invest(data, states=states)

        print(table, '->', agent.ror_history[-1])
        result[agent.ror_history[-1][0]] = table
        if show_graph:
            plt.plot(agent.ror_history)
            legend.append(table)

    if show_graph:
        plt.legend(legend, loc="upper left")
        plt.show()

    top5 = sorted(result.keys())[::-1][:5]
    for key in result.keys():
        if key not in top5:
            print('removed: ', result[key], '->', key)
            os.remove('qtables_short/' + result[key])
        else:
            print('TOP5: ', result[key], '->', key)


def load_states(data, window):
    pct = data.pct_change().as_matrix()
    bench = data.pct_change().cumsum().as_matrix()
    data_1 = pd.DataFrame(pct)
    mean = data_1.rolling(window).mean().as_matrix()
    median = data_1.rolling(window).median().as_matrix()
    std = data_1.rolling(window).std().as_matrix()
    upperbb = mean + (2 * std)
    lowerbb = mean - (2 * std)
    states = State(30, np.round(pct, ROUNDING_FACTOR), np.round(bench, ROUNDING_FACTOR),
                   np.round(mean, ROUNDING_FACTOR), np.round(median, ROUNDING_FACTOR),
                   np.round(lowerbb, ROUNDING_FACTOR), np.round(upperbb, ROUNDING_FACTOR))

    return states


if __name__ == "__main__":
    for _ in range(3):
        learn_agent()
    test_agent(show_graph=True)
    # for _ in range(10):
    #     learn_agent()
    #     test_agent()
