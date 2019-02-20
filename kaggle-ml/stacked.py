import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from scipy.stats import skew
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, \
    RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, make_scorer

import keras
from keras import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, Imputer, StandardScaler
import sklearn
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, BayesianRidge, LinearRegression, RidgeCV, LassoLarsCV, ElasticNet, \
    ElasticNetCV, OrthogonalMatchingPursuitCV, ARDRegression, LogisticRegression, LogisticRegressionCV, SGDRegressor, \
    PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
import os
import sys
import warnings
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
import lightgbm as lgb
from mlxtend.regressor import StackingRegressor

print(os.listdir("data"))


def get_cat_cols(df):
    return [col for col in df.columns if df[col].dtype == 'object']


def rmsle_cv(model, x, y):
    kf = KFold(10, shuffle=True, random_state=1).get_n_splits(x)
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kf, verbose=0))
    return (rmse)


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

y = np.log1p(train_data.SalePrice)
# test is meant for predictions and doesn't contain any price data. I need to provide it.
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
    cand_train_predictors[col] = encoders[col].transform(cand_train_predictors[col])
    cand_test_predictors[col] = encoders[col].transform(cand_test_predictors[col])

cand_train_predictors.fillna(cand_train_predictors.mean(), inplace=True)
cand_test_predictors.fillna(cand_test_predictors.mean(), inplace=True)

skewed_feats = cand_train_predictors.apply(lambda x: skew(x))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

print('Skewed features:', skewed_feats)

cand_train_predictors[skewed_feats] = np.log1p(cand_train_predictors[skewed_feats])
cand_test_predictors[skewed_feats] = np.log1p(cand_test_predictors[skewed_feats])

corr_matrix = cand_train_predictors.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
cols_to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print('Highly correlated features(will be droped):', cols_to_drop)

cand_train_predictors = cand_train_predictors.drop(cols_to_drop, axis=1)
cand_test_predictors = cand_test_predictors.drop(cols_to_drop, axis=1)

print(cand_train_predictors.shape)
print(cand_test_predictors.shape)

train_set, test_set = cand_train_predictors.align(cand_test_predictors, join='left', axis=1)

# with standard scaler
# 29  1.256766e-01            StackingRegressor
# 23  1.258170e-01                 XGBRegressor
# 17  1.259595e-01    GradientBoostingRegressor
# 24  1.301398e-01                LGBMRegressor
# 13  1.320457e-01               HuberRegressor
# 2   1.324336e-01                      LassoCV
# 6   1.324953e-01                 ElasticNetCV
# 3   1.325526e-01                  LassoLarsCV

#without standard scaler
# 17  1.255518e-01    GradientBoostingRegressor
# 23  1.259123e-01                 XGBRegressor
# 29  1.267877e-01            StackingRegressor
# 24  1.314070e-01                LGBMRegressor
# 30  1.328584e-01            StackingRegressor
# 3   1.329795e-01                  LassoLarsCV
# 4   1.330950e-01                      RidgeCV
# 5   1.331974e-01             LinearRegression
# 0   1.333531e-01                BayesianRidge
# 16  1.358579e-01          ExtraTreesRegressor
# 8   1.370994e-01                ARDRegression
# 2   1.390608e-01                      LassoCV
# 18  1.395280e-01        RandomForestRegressor
# 15  1.395452e-01             BaggingRegressor


def test_regressor(regressor, df):
    print('-' * 80)
    print(regressor)
    # my_model = make_pipeline(StandardScaler(), regressor)
    my_model = make_pipeline(regressor)
    print('Cross validated rmse score:')
    score = rmsle_cv(my_model, train_set, y)
    print(score)
    print(score.mean())
    print(score.std())
    return df.append({'score': score.mean(), 'estimator': regressor.__class__.__name__}, ignore_index=True)


df = pd.DataFrame(columns=['score', 'estimator'])

df = test_regressor(BayesianRidge(compute_score=True, copy_X=True), df)
df = test_regressor(MLPRegressor(random_state=1,
                                 activation='logistic',
                                 solver='sgd',
                                 learning_rate='adaptive',
                                 learning_rate_init=0.013000000000000001,
                                 early_stopping=True,
                                 hidden_layer_sizes=(140, 140),
                                 max_iter=10000,
                                 momentum=0.9697272727272728), df)
df = test_regressor(LassoCV(cv=5), df)
df = test_regressor(LassoLarsCV(cv=5), df)
df = test_regressor(RidgeCV(cv=5), df)
df = test_regressor(LinearRegression(), df)
df = test_regressor(ElasticNetCV(cv=5), df)
df = test_regressor(OrthogonalMatchingPursuitCV(cv=5), df)
df = test_regressor(ARDRegression(compute_score=True, copy_X=True), df)
# test_regressor(LogisticRegressionCV(cv=5)) - it's used for classification
df = test_regressor(SGDRegressor(), df)
df = test_regressor(PassiveAggressiveRegressor(), df)
df = test_regressor(RANSACRegressor(), df)
df = test_regressor(TheilSenRegressor(copy_X=True), df)
df = test_regressor(HuberRegressor(), df)
df = test_regressor(AdaBoostRegressor(n_estimators=1000), df)
df = test_regressor(BaggingRegressor(n_estimators=1000), df)
df = test_regressor(ExtraTreesRegressor(n_estimators=1000), df)
df = test_regressor(GradientBoostingRegressor(n_estimators=1000), df)
df = test_regressor(RandomForestRegressor(n_estimators=1000), df)
df = test_regressor(GaussianProcessRegressor(), df)
# df = test_regressor(IsotonicRegression(), df) - has errors
df = test_regressor(LinearSVR(), df)
df = test_regressor(NuSVR(), df)
df = test_regressor(SVR(), df)
df = test_regressor(XGBRegressor(n_estimators=1000), df)

df = test_regressor(lgb.LGBMRegressor(n_estimators=1000), df)
df = test_regressor(
    CatBoostRegressor(n_estimators=1000), df)
df = test_regressor(DecisionTreeRegressor(max_depth=3), df)
df = test_regressor(KNeighborsRegressor(), df)
# df = test_regressor(RadiusNeighborsRegressor(), df) - also has errors
df = test_regressor(DummyRegressor(), df)

df = test_regressor(StackingRegressor(
    regressors=[GradientBoostingRegressor(n_estimators=1000),
                HuberRegressor(),
                RidgeCV(cv=5),
                BayesianRidge(compute_score=True, copy_X=True)],
    meta_regressor=LassoCV(cv=5)), df)

df = test_regressor(StackingRegressor(
    regressors=[ElasticNetCV(),
                HuberRegressor(),
                RidgeCV(cv=5),
                BayesianRidge(compute_score=True, copy_X=True),
                ARDRegression(),
                MLPRegressor(random_state=1,
                             activation='logistic',
                             solver='sgd',
                             learning_rate='adaptive',
                             learning_rate_init=0.013000000000000001,
                             early_stopping=True,
                             hidden_layer_sizes=(140, 140),
                             max_iter=10000,
                             momentum=0.9697272727272728)],
    meta_regressor=LassoCV(cv=5)), df)

df.sort_values(['score'], inplace=True)

print(df)

# using the best in their class to stackem
# stacked_regressor = StackingRegressor(
#     regressors=[ElasticNetCV(),
#                 HuberRegressor(),
#                 RidgeCV(cv=5),
#                 BayesianRidge(compute_score=True, copy_X=True),
#                 ARDRegression(),
#                 MLPRegressor(random_state=1,
#                              activation='logistic',
#                              solver='sgd',
#                              learning_rate='adaptive',
#                              learning_rate_init=0.013000000000000001,
#                              early_stopping=True,
#                              hidden_layer_sizes=(140, 140),
#                              max_iter=10000,
#                              momentum=0.9697272727272728)],
#     meta_regressor=LassoCV(cv=5))
#
# # stacked_model = make_pipeline(StandardScaler(), stacked_regressor)
# stacked_model = make_pipeline(stacked_regressor)
#
# print(stacked_model)
#
# stacked_model.fit(train_set, y)
#
# gdr = GradientBoostingRegressor(n_estimators=1000)
# gdr_model = make_pipeline(StandardScaler(), gdr)
# # gdr_model = make_pipeline(gdr)
# print(gdr_model)
# gdr_model.fit(train_set, y)
#
# print('score stacked:', stacked_model.score(train_set, y))
# print('score gradient boost:', gdr_model.score(train_set, y))
#
# stacked_train_pred = stacked_model.predict(train_set)
# gdr_train_pred = gdr_model.predict(train_set)
#
# train_pred = 0.3 * stacked_train_pred + 0.7 * gdr_train_pred
#
# print('rmse from log: ', np.sqrt(mean_squared_error(y, train_pred)))
# print('mse from log: ', mean_squared_error(y, train_pred))
# print('rmsle: ', np.sqrt(mean_squared_log_error(y, train_pred)))
# print('rmse: ', np.sqrt(mean_squared_error(train_data.SalePrice, np.expm1(train_pred))))
# print('mse: ', mean_squared_error(train_data.SalePrice, np.expm1(train_pred)))
# print('mae: ', mean_absolute_error(train_data.SalePrice, np.expm1(train_pred)))
#
# stacked_test_pred = stacked_model.predict(test_set)
# gdr_test_pred = gdr_model.predict(test_set)
# test_pred = 0.3 * stacked_test_pred + 0.7 * gdr_test_pred
#
# predicted_prices = np.expm1(test_pred)
# print(predicted_prices[:5])
#
# # my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# # my_submission.Id = my_submission.Id.astype(int)
# # my_submission.to_csv('submission.csv', index=False)
