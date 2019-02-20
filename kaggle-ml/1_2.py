import pandas as pd
from sklearn.tree import DecisionTreeRegressor

pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data_file = 'data/melb_data.csv'

df = pd.read_csv(data_file)

print(df.describe())
print(df.columns)

y = df.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[features]
print(X.describe())
print(X.head())

dt = DecisionTreeRegressor(random_state=1)
dt.fit(X, y)

print('predicting price for these 5 houses:')
print(X.head(5))
print('predictions:')
predicted = dt.predict(X.head(5))
print(predicted)
print(dt.score(X.head(5), y.head(5)))
