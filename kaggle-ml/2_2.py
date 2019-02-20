import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def get_mae(X, y):
    return -1 * cross_val_score(RandomForestRegressor(50), X, y, scoring='neg_mean_absolute_error').mean()


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
numeric_cols = [col for col in train_data.columns if train_data[col].dtype != 'object']
train_data[numeric_cols] = SimpleImputer().fit_transform(train_data[numeric_cols])
numeric_cols_test = [col for col in test_data.columns if test_data[col].dtype != 'object']
test_data[numeric_cols_test] = SimpleImputer().fit_transform(test_data[numeric_cols_test])


target = train_data.SalePrice

print(train_data.describe())
print(train_data.columns)
print('-' * 80)
cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

cand_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
cand_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)
numeric_cols = [col for col in cand_train_predictors.columns if train_data[col].dtype != 'object']

low_cardinality = [col for col in cand_train_predictors if
                   train_data[col].nunique() < 10 and train_data[col].dtype == 'object']
print(low_cardinality)
my_cols = numeric_cols + low_cardinality
print(my_cols)

print('-' * 80)

train_predictors = cand_train_predictors[my_cols]
test_predictors = cand_test_predictors[my_cols]

print(train_predictors.dtypes.sample(10))

print('-' * 80)
one_hot_ecoded_train_predictors = pd.get_dummies(train_predictors)
print(one_hot_ecoded_train_predictors.columns)
print('-' * 80)
print(one_hot_ecoded_train_predictors['Condition1_Artery'])
print(train_data['Condition1'].unique())
print('-' * 80)
print('-' * 80)

predictors_without_categorical = train_predictors.select_dtypes(exclude=['object'])
mae_without_cat = get_mae(predictors_without_categorical, target)
mae_one_hot_encoded = get_mae(one_hot_ecoded_train_predictors, target)

print('mae w/o categoricals:', mae_without_cat)
print('mae one hot encoded:', mae_one_hot_encoded)
print('-' * 80)
one_hot_ecoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_ecoded_train_predictors.align(one_hot_ecoded_test_predictors,
                                                                join='left',
                                                                axis=1)

regr = RandomForestRegressor(50)
regr.fit(one_hot_ecoded_train_predictors, target)
pred = regr.predict(one_hot_ecoded_test_predictors)
print(pred)