import numpy as np
import pandas as x
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from seaborn import kdeplot
import sys

from sklearn.linear_model import Ridge

from reinforcement_learning.crypto_market.geometric_median import geometric_median
from reinforcement_learning.crypto_market.util import State

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2


def simplex_proj(y):
    """ Projection of y onto simplex. """
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1);
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    return np.maximum(y - tmax, 0.)


print('Loading price data...')
start_date = '2017-12-01'
end_date = '2018-08-01'

# tickets = ['BTC-EUR', 'ETH-BTC', 'LTC-BTC']
# tickets = ['ETH-BTC', 'LTC-BTC']
# tickets = ['BTC-EUR', 'ETH-EUR']
tickets = ['XS2L.MI', 'ANX.MI', 'IH2O.MI', 'SWDA.MI']
# tickets = ['XS2L.MI', 'ANX.MI']
# tickets = ['XS2L.MI',  'IH2O.MI']
# p = load_all_data_from_file2('btc_etf_data_adj_close_1.csv', start_date, end_date)
p = load_all_data_from_file2('mil_etf_data_adj_close.csv', start_date, end_date)
print(start_date, ' - ', end_date)
p = p[tickets]
p = p.reset_index(drop=True)
p.fillna(method="bfill", inplace=True)

d = len(tickets)

p_0 = p.copy()
p_0 = p_0.drop(max(p_0.index))
p_0 = p_0.reset_index(drop=True)
p_1 = p.copy()
p_1 = p_1.drop(0).reset_index(drop=True)

x = p_1 / p_0

# plt.plot(x)
# plt.legend(x.keys())
# plt.show()

b = np.full((len(x), d), 1. / d)
s = np.full((len(x), d), 1.)
omega = 30
_x = x.copy()
_x = _x.iloc[omega:]
_x.reset_index(drop=True, inplace=True)
for c in _x.keys():
    _x[c] = np.zeros(len(x) - omega)

epsilon = 5

for i in range(omega, len(p)):
    print('b=', b[i - omega])
    median = geometric_median(p.iloc[0:i], method='rf_regressor', options={'maxiter': 10000, 'tol': 1e-14})
    _x.loc[i - omega] = median / p.iloc[i]
    # median = geometric_median(p.iloc[i - omega:i], method='weiszfeld', options={'maxiter': 10000, 'tol': 1e-14})
    # _x.loc[i - omega] = median / p.iloc[i]

    xt = _x.loc[i - omega].values
    # print('_x.loc[i - omega] = ', xt)
    xt_ = np.mean(xt)
    # print("xt_=", xt_)
    d1 = np.power(np.linalg.norm(xt - xt_), 2)
    # print('d1= ', d1)
    a1 = (np.dot(b[i - omega], xt) - epsilon) / d1
    alpha = min([0, a1])
    # print('alpha=', alpha)
    b[i - omega + 1] = b[i - omega] - alpha * (xt - xt_)
    # print('normalized=', bnorm)
    b[i - omega + 1] = simplex_proj(b[i - omega + 1])
    # print('-' * 80)

_, ax = plt.subplots(len(tickets), 1)
for i in range(len(tickets)):
    ax[i].plot(_x[tickets[i]].iloc[20:].reset_index(drop=True))
    ax[i].plot(x[tickets[i]].iloc[omega + 20:].reset_index(drop=True))
    ax[i].legend([tickets[i]])
plt.show()

s = x * b

s1 = s.sum(axis=1)

s['return'] = s1
print(s)
print(s['return'].prod())

s['periodic return'] = [s['return'].iloc[0:i].prod() for i in s['return'].index]
# s['geo_median'] = []


_, ax = plt.subplots(len(tickets), 1)
for i in range(len(tickets)):
    ax[i].plot(b[:, i])
    ax[i].legend([tickets[i]])
plt.show()

plt.plot(s['return'])
plt.title('portfolio period return')
plt.show()

s['periodic return'].plot()
plt.title('portfolio cumulative wealth')
plt.show()
