import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Bidirectional, RepeatVector, Input
import matplotlib.pyplot as plt

from sklearn.manifold import MDS

timesteps = 5
input_dim = 10
x = np.random.uniform(0., 1., size=(100, timesteps, input_dim))

print('building model encoder-decoder model...')

input = Input(shape=(timesteps, input_dim))
encoder = Bidirectional(LSTM(timesteps, return_sequences=True))(input)
decoder = Bidirectional(LSTM(timesteps, return_sequences=True))(encoder)

sequence_autoencoder = Model(input, decoder)
encoder_model = Model(input, encoder)

print('compiling model...')
sequence_autoencoder.compile(optimizer='nadam', loss='mse', metrics=['accuracy'])

print(sequence_autoencoder.summary())
x_train, x_val = train_test_split(x)
x_train, x_test = train_test_split(x_train)
print('training model...')
history_o = sequence_autoencoder.fit(x_train, x_train, batch_size=32, epochs=1000, validation_data=(x_val, x_val))

print('predicting...')
pred = encoder_model.predict(x_test)
print(pred.shape)

print(x_test[0][0])
print(pred[0][0])

print('doing manigold learning...')
mds = MDS(verbose=True)

print('reshaping x...')
x_test_r = x_test.transpose(2, 0, 1).reshape(-1, x_test.shape[1])

print('transforming')
xx_test = mds.fit_transform(x_test_r)

print('transposing predicted results..')
pred_r = pred.transpose(2, 0, 1).reshape(-1, pred.shape[1])
print('transforming predicted results...')
ppred = mds.fit_transform(pred_r)

print('plotting....')
plt.plot(xx_test, 'o', color='red')
plt.plot(ppred, 'o', color='yellow')
plt.show()

plt.plot(history_o.history['acc'])
plt.show()
