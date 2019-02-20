import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data_file = 'data/melb_data.csv'

df = pd.read_csv(data_file)
df.dropna(inplace=True)

print(df.describe())
print(df.columns)

y = df.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[features]
print(X.describe())
print(X.head())

dt = DecisionTreeRegressor(random_state=1)
dt.fit(X, y)

predicted = dt.predict(X)
print('mean:', y.mean())
print('std:', y.std())
print('-' * 80)
print(mean_absolute_error(y, predicted))
print('-' * 80)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
dt = DecisionTreeRegressor(random_state=1)
dt.fit(train_X, train_y)
val_predictions = dt.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))

