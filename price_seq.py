from keras import Sequential
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
# d = standard_scaler.inverse_transform(d)

# playing with sklearn clusterings
# num_of_classes = 8
#
# kmean = KMeans(n_clusters=num_of_classes, n_jobs=4)
# pred = kmean.fit_predict(d)
#
# classes = np.unique(pred)
# print(classes)
#
# for clas in range(num_of_classes):
#     pp_ind = np.where(pred == clas)
#     plt.plot(data[pp_ind],pred[pp_ind],'o')
#
#
# plt.show()

# d = d[19:]
# d_seq = np.array(np.split(d, 93))
# print(d_seq.shape)
# result = []
# for dd in d_seq:
#     result.append(np.reshape(dd, (1, 30)))
#
# d_seq = np.array(result)
# print(d_seq.shape)

MAX_RANGE = 30

print('creating sequences of %d range' % (MAX_RANGE))

d_seq_file = 'd_seq.npy'

if not os.path.isfile(d_seq_file):
    d_seq = []
    for i in range(len(data) - MAX_RANGE):
        sub = np.empty((MAX_RANGE,))
        index = 0
        for j in range(i, len(data)):
            if index > 0 and index % MAX_RANGE == 0:
                d_seq.append(np.reshape(sub, (1, MAX_RANGE)))
                sub = np.empty((MAX_RANGE,))
                index = 0
            sub[index] = data[j]
            index += 1

    d_seq = np.array(d_seq)
    np.save(d_seq_file,d_seq)

d_seq = np.load(d_seq_file)

print(d_seq.shape)

x_train, x_test = train_test_split(d_seq)

m = Sequential()
m.add(LSTM(
        15,
        input_shape=(d_seq[0].shape[0], d_seq[0].shape[1]),
        return_sequences=True,
        dropout=0.2,
        kernel_regularizer='l2'))
m.add(LSTM(
        MAX_RANGE,
        return_sequences=True,
        kernel_regularizer='l2'))

m.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])

print(m.summary())

History = m.fit(x_train, x_train, epochs=5, validation_data=(x_test, x_test), batch_size=64, shuffle=True)

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.show()
