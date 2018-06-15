import numpy as np
from keras import Sequential
from keras.layers import Dense, GRU, RepeatVector, LSTM, Bidirectional, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics.classification import classification_report
import seaborn as sns


def get_model(input_dim):
    model = Sequential()

    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dense(256, activation=LeakyReLU()))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())
    return model


def get_model_lstm(input_shape):
    model = Sequential()

    model.add(TimeDistributed(Dense(256, activation=LeakyReLU()), input_shape=input_shape))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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


factor = 10

print('loading data...')
x = np.load('x.npy')
print(x.shape)

timesteps = x.shape[2]
x = np.nan_to_num(x)
x = np.repeat(x, factor, axis=0)

y_orig = np.load('y.npy')
y_orig = np.repeat(y_orig, factor, axis=0)
# dense input
print('reshaping data...')
x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
# x = x[:, [1, 2, 3, 4, 5]]
# print('adding noise')
# s = np.std(x, axis=0)
# z = np.column_stack((np.random.normal(0, s[i], len(x)) for i in range(len(s))))
# x = x+z

print(x.shape)
print('scaling data...')
# sd = StandardScaler(with_std=True)
# x = sd.fit_transform(x)
# sd = MinMaxScaler(feature_range=(-2., 2.))
# x = sd.fit_transform(x)

print('min:', np.min(x))
print('max:', np.max(x))

labels = ['ror','bench', 'lowerb', 'mean', 'median', 'higherb']

for i in range(x.shape[1]):
    sns.kdeplot(data=x[:, i], label=labels[i])

plt.legend()
plt.show()

# lstm input
# x = x.reshape(int(x.shape[0] / timesteps), timesteps, x.shape[1])
# print('lstm x shape = ', x.shape)

y_orig = y_orig.reshape(y_orig.shape[0] * y_orig.shape[1])
print('x:', x.shape)
print('y_orig:', y_orig.shape)

unique, counts = np.unique(y_orig, return_counts=True)
print(dict(zip(unique, counts)))

print('label binarization...')
# lb = LabelBinarizer()
# y = lb.fit_transform(y_orig)
y = y_orig



print('spliting data...')
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print('getting model...')
# model = get_model_lstm((x.shape[1],x.shape[2]))
model = get_model(x.shape[1])
# model = get_model_gru((x.shape[1], x.shape[2]))

print('training...')
history_o = model.fit(x_train,
                      y_train,
                      validation_split=0.2,
                      epochs=20,
                      batch_size=64,
                      shuffle=True)

score = model.evaluate(x_test, y_test, batch_size=64)
print('Evaluated:', score)

print('predicting...')
predicted = model.predict(x_test, batch_size=64)
# y_test_orig = lb.inverse_transform(y_test)
# y_pred = lb.inverse_transform(predicted)
#
# print(classification_report(y_test_orig, y_pred))

print(classification_report(y_test, np.argmax(predicted,axis=1)))

plt.plot(history_o.history['categorical_accuracy'])
plt.plot(history_o.history['val_categorical_accuracy'])
plt.title('acc vs validation acc')
plt.legend(['acc', 'val_acc'])
plt.show()
