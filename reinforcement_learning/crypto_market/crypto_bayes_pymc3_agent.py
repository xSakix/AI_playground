import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from seaborn import kdeplot
import sys

from sklearn.linear_model import Ridge

from reinforcement_learning.crypto_market.util import State

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

print('Loading price data...')
start_date = '2018-01-01'
end_date = '2018-05-01'

ticket = 'BTC-EUR'

data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)
print(start_date, ' - ', end_date)
data = data[[ticket]]
data = data.reset_index(drop=True)
data.fillna(method="bfill", inplace=True)

state = State(30,data)
print(state.get_state(10))

print(data.head(2))
print(data.tail(2))
print(len(data))
plt.plot(data)
plt.show()

action_map = {1: 'S', 2: 'B', 0: 'H'}

print('Loading data....')
x = np.load('/home/martin/Projects/AI_playground/reinforcement_learning/crypto_market/data_ga_periodic/x_back.npy')
y = np.load('/home/martin/Projects/AI_playground/reinforcement_learning/crypto_market/data_ga_periodic/y_back.npy')
# x = np.load('/home/martin/Projects/AI_playground/reinforcement_learning/crypto_market/data_ga_periodic/x.npy')
# y = np.load('/home/martin/Projects/AI_playground/reinforcement_learning/crypto_market/data_ga_periodic/y.npy')

print(len(x))
print(len(y))
exit(1)

if len(x.shape) > 2:
    x = x.reshape(x.shape[0], x.shape[1])
print('x shape = ', x.shape)
print('y shape = ', y.shape)

print('Data stats....')

_, ax = plt.subplots(2, 2)
kdeplot(x[:, 0], ax=ax[0, 0], label='pct')
kdeplot(x[:, 1], ax=ax[0, 1], label='mean')
kdeplot(x[:, 2], ax=ax[1, 0], label='median')
kdeplot(x[:, 3], ax=ax[1, 1], label='lowerbb')
kdeplot(x[:, 4], ax=ax[1, 1], label='upperbb')
plt.show()


def percentile_data(x, y, percent=89):
    legends = ['pct', 'mean', 'median', 'lowerbb', 'upperbb']
    for i in range(x.shape[1]):
        print('%s %d percentile = %f' % (legends[i], percent, np.percentile(x[:, i], percent)))

    print('action ', percent, ' percentile = ', action_map[np.percentile(y, percent)])


print('--percentiles---')
percentile_data(x, y)
print('---')
percentile_data(x, y, 95)
print('---')
percentile_data(x, y, 11)
print('---')
print('actions:')
unique, counts = np.unique(y, return_counts=True)
counts_dict = dict(zip(unique, counts))
for key in action_map.keys():
    print(action_map[key], ' = ', counts_dict[key])

kdeplot(y)
plt.title('actions')
plt.show()

color_map = {0: 'black', 1: 'red', 2: 'blue'}
sell_legend = mpatches.Patch(color='red', label='sell')
buy_legend = mpatches.Patch(color='blue', label='buy')

for i in range(len(x)):
    if y[i] == 0:
        continue
    plt.plot(i, x[i, 0], color=color_map[y[i]], marker='o')
plt.plot(range(len(x)), np.zeros(len(x)), color='black')
plt.grid(color='b')
plt.legend(handles=[sell_legend, buy_legend])
plt.show()

btc_legend, = plt.plot(data, color='black', label='btc-eur')
for i in range(30, len(data)):
    if y[i - 30] == 0:
        continue
    plt.plot(i, data.iloc[i], color=color_map[y[i - 30]], marker='o')
plt.plot(range(len(data)), np.zeros(len(data)), color='black')
plt.grid(color='b')
plt.legend(handles=[btc_legend, sell_legend, buy_legend])
plt.show()

reg = Ridge(alpha=1.0)
reg.fit(x, y)
print(reg.coef_)

df_coef = pd.DataFrame(reg.coef_.reshape(-1, len(reg.coef_)),
                       columns=['pct', 'mean', 'median', 'lowerbb', 'higherbb'])

print(df_coef.to_string())

# print(np.corrcoef(x))
