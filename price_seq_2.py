from keras import Sequential, Input, Model
from keras.layers import LSTM, Reshape, GRU, Dropout, Dense, Bidirectional
from pandas_datareader import data as data_reader
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
import os
import pandas as pd

data_file = 'btc.data.csv'

if not os.path.isfile(data_file):
    data = data_reader.get_data_yahoo('BTC-USD')
    data.to_csv(data_file)

data = pd.read_csv(data_file)

print(data.Open.head())
print(data.Open.tail())
print(data.Open.iloc[-1])
hpd = pm.hpd(data.Open, alpha=0.05)
print('[%f %f]' % (hpd[0], hpd[1]))

# _, (ax0, ax1) = plt.subplots(2, 1)
# sns.kdeplot(data.Open, ax=ax0)
# ax1.plot(data.Open)
# plt.show()

data = data.as_matrix(columns=['Open'])
print(data.shape)

standard_scaler = StandardScaler()
standard_scaler.fit(data)
d = standard_scaler.transform(data)

MAX_RANGE = 100

num = int(len(d) / MAX_RANGE)
start = len(d) - num * MAX_RANGE

d = d[start:]
d_seq = np.array(np.split(d, num))
print(d_seq.shape)

result = []
for dd in d_seq:
    result.append(np.reshape(dd, (1, MAX_RANGE)))

d_seq = np.array(result)
print(d_seq.shape)

x_train, x_test = train_test_split(d_seq)

print('building model encoder-decoder model...')

input = Input(shape=(d_seq.shape[1], d_seq.shape[2]))
encoder = Bidirectional(LSTM(int(MAX_RANGE/2), return_sequences=True))(input)
decoder = Bidirectional(LSTM(int(MAX_RANGE/2), return_sequences=True))(encoder)

sequence_autoencoder = Model(input, decoder)
encoder_model = Model(input, encoder)

print('compiling model...')
sequence_autoencoder.compile(optimizer='nadam', loss='mse', metrics=['accuracy'])

history_o = sequence_autoencoder.fit(x_train, x_train, epochs=50000, validation_split=0.3, batch_size=64, shuffle=True)

plt.plot(history_o.history['acc'])
plt.show()

prediction = encoder_model.predict(d_seq)
print(prediction.shape)

pred = []
for val in prediction:
    pred.append(val.reshape(MAX_RANGE, ))

pred = np.array(pred)
pred = np.concatenate(pred)
print(pred.shape)

_, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(d)
ax1.plot(pred)
plt.show()
