from keras import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Bidirectional
from keras.optimizers import Adam, Nadam


def create_lstm_model(learning_rate, timesteps):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.3, return_sequences=True), input_shape=(timesteps, 5)))
    model.add(BatchNormalization())
    model.add(LSTM(32))
    model.add(Dense(3, activation='softmax'))

    opt = Nadam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model


def create_dense_model(learning_rate):
    model = Sequential()
    model.add(Dense(128, activation='tanh', input_dim=5))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    # model.add(Dense(32, activation='relu', input_dim=5))
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax'))
    opt = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model
