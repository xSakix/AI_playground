import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('data/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.fit_transform(test_X)

from xgboost import XGBRegressor

my_model = XGBRegressor(random_state=1)
print(my_model)
my_model.fit(train_X, train_y)

predictions = my_model.predict(test_X)
from sklearn.metrics import mean_absolute_error

print('mae: ', mean_absolute_error(test_y, predictions))

my_model = XGBRegressor(n_estimators=100000, random_state=1)
my_model.fit(train_X, train_y,
             early_stopping_rounds=5,
             eval_set=[(test_X, test_y)],
             verbose=False)

predictions = my_model.predict(test_X)

print('mae: ', mean_absolute_error(test_y, predictions))

my_model = XGBRegressor(n_estimators=10000, learning_rate=0.05, random_state=1)
my_model.fit(train_X, train_y,
             early_stopping_rounds=5,
             eval_set=[(test_X, test_y)],
             verbose=False)

predictions = my_model.predict(test_X)

print('mae: ', mean_absolute_error(test_y, predictions))

# for estimators in range(200, 1000, 50):
#     my_model = XGBRegressor(n_estimators=estimators, random_state=1)
#     my_model.fit(train_X, train_y,
#                  early_stopping_rounds=5,
#                  eval_set=[(test_X, test_y)],
#                  verbose=False)
#
#     predictions = my_model.predict(test_X)
#
#     print('mae[%d]:[%f] ' % (estimators, mean_absolute_error(test_y, predictions)))

steps = np.linspace(0.001, 0.1, 100)
maes = []

for learning_rate in steps:
    my_model = XGBRegressor(n_estimators=1000, learning_rate=learning_rate, random_state=1)
    my_model.fit(train_X, train_y,
                 early_stopping_rounds=5,
                 eval_set=[(test_X, test_y)],
                 verbose=False)

    predictions = my_model.predict(test_X)
    maes.append(mean_absolute_error(test_y, predictions))
    # print('mae[%f]:[%f] ' % (learning_rate, maes[-1]))

import matplotlib.pyplot as plt

index = np.argmin(maes)
# print('min at ', steps[index])

plt.plot(steps, maes)
plt.show()

# print('-'*80)
# 0.047000
learning_rate = steps[index]
my_model = XGBRegressor(n_estimators=1000, learning_rate=learning_rate, random_state=1)
my_model.fit(train_X, train_y,
             early_stopping_rounds=5,
             eval_set=[(test_X, test_y)],
             verbose=False)

print(my_model.feature_importances_)

predictions = my_model.predict(test_X)
print('mae[%f]:[%f] ' % (learning_rate, mean_absolute_error(test_y, predictions)))
