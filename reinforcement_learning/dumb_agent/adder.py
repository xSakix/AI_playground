import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

np.random.seed(2)

x = np.random.randint(0, 100, (130000, 2))
y = x[:, 0] + x[:, 1]

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=2))
# model.add(Dropout(0.1))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='nadam', loss='mse')
#
history = model.fit(x, y,
                    epochs=20,
                    validation_split=0.3,
                    shuffle=True,
                    callbacks=[EarlyStopping()])

x_test = np.random.randint(0, 100, (100, 2))
y_test = x_test[:, 0] + x_test[:, 1]

score = model.evaluate(x_test, y_test)

print(score)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

y_pred = model.predict(x_test)
for i in range(len(x_test)):
    print(x_test[i, 0], '+', x_test[i, 1], '=', y_test[i], '|', np.rint(y_pred[i]))
