import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

data = pd.read_csv('data/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['Id', 'SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.fit_transform(test_X)

learning_rate = 0.047000
my_model = XGBRegressor(n_estimators=1000, learning_rate=learning_rate, random_state=1)
print(my_model)
my_model.fit(train_X, train_y,
             early_stopping_rounds=5,
             eval_set=[(test_X, test_y)],
             verbose=False)

print(my_model.feature_importances_)
print(X.columns)

sorted_args = reversed(np.argsort(my_model.feature_importances_))

for i in sorted_args:
    print(X.columns[i], '->', my_model.feature_importances_[i])

predictions = my_model.predict(test_X)
print('mae[%f]:[%f] ' % (learning_rate, mean_absolute_error(test_y, predictions)))


def get_some_data(cols_to_use):
    data = pd.read_csv('data/train.csv')
    y = data.SalePrice
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y


sorted_args = reversed(np.argsort(my_model.feature_importances_))
cols_to_use = []
for i in sorted_args:
    cols_to_use.append(X.columns[i])
    if len(cols_to_use) > 10:
        break

print(cols_to_use)
# get_some_data is defined in hidden cell above.
X, y = get_some_data(cols_to_use)

my_model = GradientBoostingRegressor()
# fit the model as usual
my_model.fit(X, y)
print(my_model.feature_importances_)
# Here we make the plot
my_plots = plot_partial_dependence(my_model,
                                   features=[0, len(cols_to_use)-1],  # column numbers of plots we want to show
                                   X=X,  # raw predictors data.
                                   feature_names=cols_to_use,  # labels on graphs
                                   grid_resolution=10)  # number of values to plot on x axis
plt.show()
