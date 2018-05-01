import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Flatten, RepeatVector
import matplotlib.pyplot as plt
from sklearn.metrics.classification import classification_report

classes = 100000
timesteps = 5
sequence = 10

num_of_classification_classes = 100

print('generating data...')
x = np.random.uniform(0.5, 1.5, size=(classes, timesteps, sequence))
y = np.random.randint(0, num_of_classification_classes, size=(classes, ))

print(x.shape)
print(y.shape)

print('loading encoder...')
model_encoder = load_model('encoder.h5')
print('encoding...')
x = model_encoder.predict(x)

print('creating train, test,val data sets...')
x_train, x_test, y_train, y_test = train_test_split(x, y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

print('building model...')
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(timesteps, sequence)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(num_of_classification_classes, activation='softmax'))

print('compiling model...')
model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['acc'])
print(model.summary())

print('fitting model...')
history_o = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=500, batch_size=64)

print('evaluating model...')
score = model.evaluate(x_test, y_test, batch_size=64)
print(score)

plt.plot(history_o.history['acc'])
plt.show()

print('predicting classes')
pred = model.predict(x_test, batch_size=64)

# print(np.argmax(pred, axis=1))

print('printing report...')
print(classification_report(y_test, np.argmax(pred, axis=1)))
