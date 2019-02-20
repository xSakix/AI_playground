import multiprocessing
import threading

import numpy as np
from reinforcement_learning.crypto_market.util import my_score
import time


def create_actions(length, r_actions=None):
    if r_actions is None:
        return np.random.randint(0, 3, size=length)
    else:
        return r_actions


def run(data, states, invested=100000, agent=None):
    ror_history = np.full(len(data) - states.window, fill_value=np.nan)
    r_actions = create_actions(len(data) - states.window)
    state = []
    cash = invested
    shares = 0
    score = 0

    for i in range(states.window, len(data)):
        prices = data.iloc[i].values
        portfolio = cash + np.dot(prices, shares)
        action = r_actions[i - states.window]
        action, cash, share = do_action(action, portfolio, prices, cash, shares)
        r_actions[i - states.window] = action
        state.append(states.get_state(i))
        ror_history[i - states.window] = (portfolio - invested) / invested

    ror_history = np.nan_to_num(np.array(ror_history))

    try:
        ror_agent = np.nan_to_num(np.copy(agent.ror_history))
        score += my_score(ror_history, ror_agent)
    except:
        score += my_score(ror_history, states.bench)

    state = np.array(state)
    return state, ror_history, score


def do_action(action, portfolio, prices, cash, shares, coef=0.5, tr_cost=0.003):
    if action == 1:
        return sell(prices, cash, shares, coef, tr_cost)
    elif action == 2:
        return buy(portfolio, prices, cash, shares, coef, tr_cost)
    else:
        return 0, cash, shares


def buy(portfolio, prices, cash, shares, coef=0.5, tr_cost=0.003):
    available_cash = cash * coef
    cost = tr_cost* available_cash
    available_cash = available_cash - cost

    if available_cash < 0.001 * prices:
        return 0, cash, shares

    shares += available_cash/prices
    cash = portfolio - shares* prices - cost

    return 2, cash, shares


def sell(prices, cash, shares, coef=0.5, tr_cost=0.003):
    to_sell = coef * shares
    if to_sell == 0:
        return 0, cash, shares

    sold = to_sell*prices
    cash += sold - sold * tr_cost
    shares = shares - to_sell

    return 1, cash, shares


def score_based_on_ror(ror):
    if ror > 0.:
        return 1
    return 0


def score_based_on_beating_benchmark(ror, benchmark):
    if ror > 0. and ror >= benchmark:
        return 1 + int(ror / benchmark)
    return 0


def score_based_on_beating_trader(ror, ror_agent):
    if ror > 0. and ror >= ror_agent:
        try:
            return 1 + int(ror / ror_agent)
        except:
            return 1


if __name__ == "__main__":
    from reinforcement_learning.crypto_market.util import State
    import sys
    import matplotlib.pyplot as plt
    import time
    sys.path.insert(0, '../../../etf_data')
    from etf_data_loader import load_all_data_from_file2

    start_date = '2017-05-01'
    end_date = '2018-05-01'

    ticket = 'BTC-EUR'
    window = 30
    data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)
    print(start_date, ' - ', end_date)
    data = data[[ticket]].reset_index(drop=True).fillna(method="bfill")
    print(data.head(2))
    print(data.tail(2))

    states = State(window, data)

    t1 = time.time()
    for _ in range(100):
        state, ror_history, score = run(data,states)
    print('time:', time.time() - t1)

    print(len(state))
    print(score)

    plt.plot(ror_history)
    plt.show()
