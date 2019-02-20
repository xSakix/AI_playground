import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def score_dataset(train_X, test_X, train_y, test_y):
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    return mean_absolute_error(test_y, preds)


data_file = 'data/house_pricing_snapshot/melb_data.csv'
df = pd.read_csv(data_file)

print(df.describe())
print(df.columns)
print('-' * 80)
missing_vals = df.isnull().sum()
print(missing_vals)
print('-' * 80)
print(missing_vals[missing_vals > 0])
print('-' * 80)

melb_target = df.Price
melb_predictors = df.drop(['Price'], axis=1)
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
print(melb_numeric_predictors.head())
print('Shapes:')
print(melb_target.shape)
print(melb_numeric_predictors.shape)

train_X, test_X, train_y, test_y = train_test_split(melb_numeric_predictors,
                                                    melb_target,
                                                    test_size=0.3,
                                                    train_size=0.7,
                                                    random_state=0)
print('-' * 80)
print('Shapes of train/test sets:')
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)

# nemozem lebo chybaju hodnoty
# print('-' * 80)
# print('score original:')
# print(score_dataset(train_X, test_X, train_y, test_y))

cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]
print('-' * 80)
print('missing cols:')
print(cols_with_missing)

print('-' * 80)
reduced_train_X = train_X.drop(cols_with_missing, axis=1)
reduced_test_X = test_X.drop(cols_with_missing, axis=1)
print(score_dataset(reduced_train_X, reduced_test_X, train_y, test_y))

print('-' * 80)
imputed_train_X_plus = train_X.copy()
imputed_test_X_plus = test_X.copy()

for col in cols_with_missing:
    imputed_train_X_plus[col+'_was_missing'] = imputed_train_X_plus[col].isnull()
    imputed_test_X_plus[col+'_was_missing'] = imputed_test_X_plus[col].isnull()

my_imputer = SimpleImputer()
imputed_train_X_plus = my_imputer.fit_transform(imputed_train_X_plus)
imputed_test_X_plus = my_imputer.fit_transform(imputed_test_X_plus)
print(score_dataset(imputed_train_X_plus, imputed_test_X_plus, train_y, test_y))

