import os

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import MSE
from sklearn.preprocessing import LabelBinarizer

from reinforcement_learning.crypto_market.crypto_random_agent import CryptoRandomAgent, State
import sys

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time

import keras.backend as K

np.warnings.filterwarnings('ignore')




start_date = '2017-01-01'
end_date = '2018-07-01'

ticket = 'BTC-EUR'
window = 30

data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)
print(start_date, ' - ', end_date)
data = data[[ticket]]
data = data.reset_index(drop=True)
data.fillna(method="bfill", inplace=True)
print(data.head(2))
print(data.tail(2))
print(len(data))

pct = data.pct_change().fillna(method='bfill')
bench = pct.cumsum().as_matrix()
pct = pct.as_matrix().reshape(-1)

data_1 = pd.DataFrame(pct)
mean = data_1.rolling(window=window).mean().fillna(method='bfill').as_matrix().reshape(-1)
median = data_1.rolling(window=window).median().fillna(method='bfill').as_matrix().reshape(-1)
std = data_1.rolling(window=window).std().fillna(method='bfill').as_matrix().reshape(-1)
upperbb = mean + (2 * std)
lowerbb = mean - (2 * std)

states = State(window,pct,bench,mean,median,lowerbb,upperbb)

print(pct.shape)
print(mean.shape)
print(median.shape)
print(lowerbb.shape)
print(upperbb.shape)
x = np.stack([pct, lowerbb, mean, median, upperbb], axis=1)
print(x.shape)


model = Sequential()
model.add(Dense(int(2 * x.shape[1] + 1), input_dim=x.shape[1]))
model.add(Dense(3, activation='sigmoid'))


def agent_eval(y_true, y_pred):
    print(y_pred.shape)
    actions = K.argmax(y_pred)
    agent = CryptoRandomAgent(ticket, r_actions=actions)
    agent.invest(data,states)
    return K.subtract(1.,agent.score)


model.compile(optimizer='rmsprop', loss=agent_eval, metrics=['acc'])
print(model.summary())

history = model.fit(x, x, validation_split=0.1, callbacks=[EarlyStopping()], shuffle=True, epochs=100)
print(model.evaluate(x, x))
