'''
this one will use sequences generated by create_sequences.py

'''

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

print('loading data...')

data_file = 'btc-seq.csv'
data = pd.read_csv(data_file)

print('creating input matrix....')
orig_data = data.as_matrix()
print(data.shape)

print('scaling data...')
standard_scaler = StandardScaler()
standard_scaler.fit(orig_data)
d = standard_scaler.transform(orig_data)
data = d.reshape(d.shape[0], 1, d.shape[1])
print(data.shape)

x_train, x_test = train_test_split(data,test_size=0.2)

print('building model encoder-decoder model...')

input = Input(shape=(data.shape[1], data.shape[2]))
encoder = Bidirectional(LSTM(
                int(data.shape[2]/2),
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.3))(input)
decoder = Bidirectional(LSTM(
                int(data.shape[2]/2),
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.2))(encoder)

sequence_autoencoder = Model(input, decoder)
encoder_model = Model(input, encoder)

print('compiling model...')
sequence_autoencoder.compile(optimizer='nadam', loss='mse', metrics=['accuracy'])
print(sequence_autoencoder.summary())

print('fitting model...')
history_o = sequence_autoencoder.fit(
        x_train,
        x_train,
        epochs=2000,
        validation_split=0.1,
        batch_size=64,
        shuffle=True)

plt.plot(history_o.history['acc'])
plt.plot(history_o.history['loss'])
plt.show()

print('evaluating...')
score = sequence_autoencoder.evaluate(x_test, x_test)
print('acc:%f'%score[1])

print('predicting....')
prediction = encoder_model.predict(data)
print(prediction.shape)

print('deconstructing to timeseries...')
pred = prediction.reshape(prediction.shape[0], prediction.shape[2])
pred = pred.flatten()
print(pred.shape)

flatten_data = orig_data.flatten()

_, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(flatten_data)
ax1.plot(pred)
plt.show()
