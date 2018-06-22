import os

import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, GRU, RepeatVector, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.layers.advanced_activations import LeakyReLU
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics.classification import classification_report
import seaborn as sns


def get_model(input_dim):
    model = Sequential()

    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax', kernel_regularizer='l1_l2'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    return model


def get_model_lstm(input_shape):
    model = Sequential()

    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer='nadam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    print(model.summary())
    return model


def get_model_gru(input_shape):
    model = Sequential()

    model.add(Bidirectional(GRU(256), input_shape=input_shape))
    model.add(Dense(256, activation=LeakyReLU()))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())
    return model


print('loading data...')
x_orig = np.load('x.npy')
x = x_orig
print(x.shape)

if len(x.shape) == 3:
    timesteps = x.shape[2]
else:
    timesteps = 1

x = np.nan_to_num(x)

y_orig = np.load('y.npy')
print('reshaping data...')
if len(x.shape) == 3:
    x = x[:, :, 2:7]
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
elif (len(x.shape) == 4):
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
else:
    x = x[:, 2:7]

if os.path.isfile('keras_model_eu/standard_scaler.pkl'):
    scaler = joblib.load('keras_model_eu/standard_scaler.pkl')
else:
    scaler = StandardScaler()
    scaler.fit(x)
    joblib.dump(scaler, 'keras_model_eu/standard_scaler.pkl')

x = scaler.transform(x)
# print(x.shape)
# x = x.reshape(x.shape[0], 1, x.shape[1])

print(x.shape)

print('min:', np.min(x))
print('max:', np.max(x))

if len(y_orig.shape) == 2:
    y_orig = y_orig.reshape(y_orig.shape[0] * y_orig.shape[1])

print('x:', x.shape)
print('y_orig:', y_orig.shape)

unique, counts = np.unique(y_orig, return_counts=True)
print(dict(zip(unique, counts)))

if os.path.isfile('keras_model_eu/label_bin.pkl'):
    lbl = joblib.load('keras_model_eu/label_bin.pkl')
else:
    lbl = LabelBinarizer(sparse_output=False)
    y = lbl.fit(y_orig)
    joblib.dump(lbl, 'keras_model_eu/label_bin.pkl')

y = lbl.transform(y_orig)
# y = y_orig

print('spliting data...')
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print('getting model...')
# model = get_model_lstm((x.shape[1], x.shape[2]))
model = get_model(x.shape[1])
# model = get_model_gru((x.shape[1], x.shape[2]))

print('training...')
history_o = model.fit(x_train,
                      y_train,
                      validation_split=0.2,
                      epochs=100,
                      batch_size=64,
                      shuffle=True,
                      callbacks=[EarlyStopping()])

score = model.evaluate(x_test, y_test, batch_size=64)
print('Evaluated:', score)

print('predicting...')
predicted = model.predict(x_test, batch_size=64)
predicted = lbl.inverse_transform(predicted)
y_test = lbl.inverse_transform(y_test)

# print(classification_report(y_test, np.argmax(predicted, axis=1)))
print(classification_report(y_test, predicted))

plt.plot(history_o.history['acc'])
plt.plot(history_o.history['val_acc'])
# plt.plot(history_o.history['categorical_accuracy'])
# plt.plot(history_o.history['val_categorical_accuracy'])
plt.title('acc vs validation acc')
plt.legend(['acc', 'val_acc'])
plt.show()

id = len(os.listdir('keras_model_eu'))

model.save('keras_model_eu/mlpp_' + str(id) + '.h5')
