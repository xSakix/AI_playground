import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
#     dt = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     dt.fit(train_X, train_y)
#     val_predictions = dt.predict(val_X)
#     return mean_absolute_error(val_y, val_predictions)


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

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# for max_leaf_nodes in [5, 50, 500, 5000, 50000]:
#     mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print(max_leaf_nodes, ' -> ', mae)
model = RandomForestRegressor(random_state=1)
model.fit(train_X, train_y)
y_pred = model.predict(val_X)
print(mean_absolute_error(val_y, y_pred))

