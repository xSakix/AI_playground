import pandas as pd

import numpy as np
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_lstm(input_shape, model_src):
    model = Sequential()

    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(3, activation='softmax'))

    model.load_weights(model_src)

    return model


def load_mlp_model(input_dim, model_src):
    model = Sequential()

    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.load_weights(model_src)
    return model

def load_mlpp_model(input_dim, model_src):
    model = Sequential()

    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax', kernel_regularizer='l1_l2'))

    model.load_weights(model_src)
    return model



def load_scaler(src):
    return joblib.load(src)


class CryptoTraderAgent:
    def __init__(self, ticket, model='models/decision_tree.pkl', invested=100000.,
                 scaler='keras_model_eu/standard_scaler.pkl',
                 binarizer=None):
        self.r_actions = []
        self.score = 0
        self.ror_history = []
        self.dist = np.array([1.])
        self.ticket = ticket
        self.tr_cost = 0.003
        self.invested = invested
        self.cash = invested
        self.shares = np.zeros(1)
        self.history = []
        self.actions = []
        self.state = []
        self.model = model

        self.iskeras = False
        self.ismlp = False
        self.scaler = None
        self.binarizer = None

        if model.startswith('keras') and model.__contains__('lstm'):
            self.clf = load_lstm((1, 5), model_src=model)
            self.iskeras = True
            self.scaler = load_scaler(scaler)

        if model.startswith('keras') and model.__contains__('mlp'):
            if model.__contains__('mlpp'):
                self.clf = load_mlpp_model(5, model_src=model)
                self.binarizer = joblib.load(binarizer)
            else:
                self.clf = load_mlp_model(5, model_src=model)
            self.scaler = load_scaler(scaler)
            self.iskeras = True
            self.ismlp = True

        if model.startswith('models'):
            self.clf = joblib.load(model)
            self.iskeras = False

        self.coef = 0.5

    def invest(self, data, window=30):

        if len(data.keys()) == 0:
            return

        data.fillna(method="bfill", inplace=True)

        pct = data.pct_change().as_matrix()
        bench = data.pct_change().cumsum().as_matrix()
        data_1 = pd.DataFrame(pct)
        mean = data_1.rolling(window=window).mean().as_matrix()
        median = data_1.rolling(window=window).median().as_matrix()
        std = data_1.rolling(window=window).std().as_matrix()
        upperbb = mean + (2 * std)
        lowerbb = mean - (2 * std)

        self.ror_history = np.empty(len(data))
        self.ror_history[:] = np.nan

        for i in range(window + 1, len(data)):

            prices = data.iloc[i].values

            portfolio = self.cash + np.dot(prices, self.shares)

            try:
                if np.isnan(portfolio):
                    portfolio = 0.
            except:
                print('portfolio:', portfolio)

            self.history.append(portfolio)

            ror = (portfolio - self.invested) / self.invested

            self.ror_history[i] = ror

            input = [[pct[i - 1],
                      lowerbb[i - 1],
                      mean[i - 1],
                      median[i - 1],
                      upperbb[i - 1]]]
            input = np.array(input)

            if self.iskeras:
                if not self.ismlp:
                    x = np.reshape(input, (1, 5))
                    x = self.scaler.transform(x)
                    if self.binarizer is not None:
                        action = self.clf.predict(np.reshape(x, (1, 1, 5)))
                        action = self.binarizer.inverse_transform(action)
                    else:
                        action = np.argmax(self.clf.predict(np.reshape(x, (1, 1, 5))), axis=1)
                else:
                    x = np.reshape(input, (1, 5))
                    x = self.scaler.transform(x)
                    action = np.argmax(self.clf.predict(x), axis=1)
            else:
                action = self.clf.predict(np.reshape(input, (1, 5)))

            self.state.append(input)
            self.r_actions.append(action)

            if action == 0:
                self.actions.append('H')
                continue
            elif action == 1 and sum(self.shares > 0) > 0:
                self.actions.append('S')
                to_sell = self.coef * self.shares
                sold = np.dot(to_sell, prices)
                self.cash += sold - sold * self.tr_cost
                self.shares = self.shares - to_sell
                # portfolio = self.cash + np.dot(prices, self.shares)
                # print('selling ', to_sell, ' portfolio=', portfolio, 'cash=', self.cash,'shares=',self.shares)
            elif action == 2 and (self.coef * self.cash - self.tr_cost * self.coef * self.cash) > 0.000000001 * prices:
                self.actions.append('B')
                c = self.cash * self.coef
                cost = np.multiply(self.tr_cost, c)
                c = np.subtract(c, cost)
                s = np.divide(c, prices)
                self.shares += s
                self.cash = portfolio - np.dot(self.shares, prices) - cost
                # portfolio = self.cash + np.dot(prices, self.shares)
                # print('buying ', s, ' portfolio=', portfolio, 'cash=', self.cash,'shares=',self.shares)
            else:
                self.actions.append('H')

        df = pd.DataFrame(self.ror_history)
        df.fillna(method="bfill", inplace=True)
        self.ror_history = df.as_matrix()
