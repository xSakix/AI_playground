import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV, cross_val_score

print(os.listdir("data"))

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


def get_cat_cols(df):
    return [col for col in df.columns if df[col].dtype == 'object']


y = np.log1p(train_data.SalePrice)
cand_train_predictors = train_data.drop(['Id', 'SalePrice'], axis=1)
cand_test_predictors = test_data.drop(['Id'], axis=1)
cat_cols = get_cat_cols(cand_train_predictors)

cand_train_predictors[cat_cols] = cand_train_predictors[cat_cols].fillna('NotAvailable')
cand_test_predictors[cat_cols] = cand_test_predictors[cat_cols].fillna('NotAvailable')

encoders = {}

for col in cat_cols:
    encoders[col] = LabelEncoder()
    val = cand_train_predictors[col].tolist()
    val.extend(cand_test_predictors[col].tolist())
    encoders[col].fit(val)
    cand_train_predictors[col] = encoders[col].transform(cand_train_predictors[col]) + 1
    cand_test_predictors[col] = encoders[col].transform(cand_test_predictors[col]) + 1

corr_matrix = cand_train_predictors.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
cols_to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print('correlated features(will be droped):', cols_to_drop)

cand_train_predictors = cand_train_predictors.drop(cols_to_drop, axis=1)
cand_test_predictors = cand_test_predictors.drop(cols_to_drop, axis=1)

print(cand_train_predictors.shape)
print(cand_test_predictors.shape)

train_set, test_set = cand_train_predictors.align(cand_test_predictors, join='left', axis=1)
train_set = np.log1p(train_set)
test_set = np.log1p(test_set)

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

params = {}

train_set.fillna('NaN', inplace=True)

score_results = []
kfold = KFold(n_splits=10, random_state=1)
imputer = SimpleImputer()
# scaler = RobustScaler(with_scaling=True, with_centering=True, quantile_range=(20., 80.))
scaler = RobustScaler()
select = SelectFromModel(LassoCV(cv=kfold, random_state=1), threshold='median')

regressor = MLPRegressor(early_stopping=True,
                         activation='identity',
                         max_iter=10000)

my_model = make_pipeline(imputer, scaler, select, regressor)
scores = np.sqrt(
    -1 * cross_val_score(my_model,
                         train_set,
                         y,
                         scoring='neg_mean_squared_log_error',
                         verbose=0,
                         n_jobs=2,
                         cv=kfold))
mean_score = scores.mean()
print(mean_score)
print(scores.std())

regressor = MLPRegressor(early_stopping=True,
                         activation='identity',
                         max_iter=10000)

my_model = make_pipeline(imputer, scaler, select, regressor)

my_model.fit(train_set, y)
print(my_model.score(train_set, y))

# sgd
# 0.010535126813418304
# 0.0014979832026646835
# 0.8753998274053919
# adam
# 0.01042968407581037
# 0.0015518828940809886
# 0.8953275894476386
# rmsle:  0.009936016739047336
# rmse:  30724.85212172329
# mae:  16180.395157871164
# [120197.07675005 158456.00818102 180513.20130843 194200.42489418
#  180857.53842551]


train_pred = my_model.predict(train_set)
print('rmsle: ', np.sqrt(mean_squared_log_error(y, train_pred)))
print('rmse: ', np.sqrt(mean_squared_error(train_data.SalePrice, np.expm1(train_pred))))
print('mae: ', mean_absolute_error(train_data.SalePrice, np.expm1(train_pred)))

test_set.fillna('NaN', inplace=True)
predicted_prices = np.expm1(my_model.predict(test_set))
print(predicted_prices[:5])

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
my_submission.Id = my_submission.Id.astype(int)
my_submission.to_csv('submission.csv', index=False)
