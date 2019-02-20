import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from scipy.stats import skew
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.model_selection.tests.test_validation import test_validation_curve_cv_splits_consistency
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, \
    RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, make_scorer

import keras
from keras import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization, LSTM
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
import seaborn as sns

print(os.listdir("data"))


def get_cat_cols(df):
    return [col for col in df.columns if df[col].dtype == 'object']


def rmsle_cv(model, x, y):
    kf = KFold(10, shuffle=True, random_state=1).get_n_splits(x)
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kf, verbose=0))
    return (rmse)


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

to_str = ['YearBuilt','LotArea','MasVnrArea','BsmtFinSF1','1stFlrSF','2ndFlrSF','LotFrontage']
# to_str = ['YearBuilt']
to_few = ['Street','Utilities','LandSlope','Condition2']

for column in train_data.columns:
    print(train_data[column].head(5))
    if column == 'Id':
        continue
    df = pd.DataFrame(columns=[column, 'SalePrice'])
    df['SalePrice'] = train_data.SalePrice

    if train_data[column].dtype != 'object':
        train_data[column] = train_data[column].fillna(train_data[column].mean())

    if column in to_str:
        plt.scatter(train_data[column], train_data.SalePrice)
        plt.xlabel(column)
        plt.ylabel('sale price')
        plt.plot(np.linspace(min(train_data[column]), max(train_data[column]), len(train_data[column])),
                 np.linspace(min(train_data.SalePrice), max(train_data.SalePrice), len(train_data[column])),
                 color='black')
        plt.show()

        if train_data[column].dtype == 'float64':
            train_data[column] = train_data[column].astype('int')
        train_data[column] = train_data[column].astype('object')
    if train_data[column].dtype == 'int64':
        plt.scatter(train_data[column], train_data.SalePrice)
        plt.xlabel(column)
        plt.ylabel('sale price')
        plt.plot(np.linspace(min(train_data[column]), max(train_data[column]), len(train_data[column])),
                 np.linspace(min(train_data.SalePrice), max(train_data.SalePrice), len(train_data[column])),
                 color='black')
        plt.show()
        train_data[column] = train_data[column].astype('object')

    if train_data[column].dtype == 'object':
        train_data[column] = train_data[column].fillna('NotAvailable')
        df[column] = LabelEncoder().fit_transform(train_data[column])
    else:
        df[column] = train_data[column]

    plt.scatter(df[column], df.SalePrice)
    plt.xlabel(column)
    plt.ylabel('sale price')
    plt.plot(np.linspace(min(df[column]), max(df[column]), len(df[column])),
             np.linspace(min(df.SalePrice), max(df.SalePrice), len(df[column])),
             color='black')
    plt.show()

exit(1)

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

pd.set_option("use_inf_as_na", True)

corr_matrix = cand_train_predictors.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
cols_to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print('Highly correlated features(will be droped):', cols_to_drop)

cand_train_predictors = cand_train_predictors.drop(cols_to_drop, axis=1)
cand_test_predictors = cand_test_predictors.drop(cols_to_drop, axis=1)

# for column in cand_train_predictors.columns:
#     print('-' * 80)
#     print(column)
#     coef = np.corrcoef(cand_train_predictors[column], train_data.SalePrice)
#     if coef[0][1] == -1.:
#         print('reciprocal')
#         cand_train_predictors[column] = np.power(cand_train_predictors[column], -1)
#     elif coef[0][1] > -1. and coef[0][1] <= -.5:
#         print('reciprocal square root')
#         cand_train_predictors[column] = np.power(cand_train_predictors[column], -1 / 2)
#     elif coef[0][1] > -.5 and coef[0][1] <= 0.0:
#         print('log')
#         cand_train_predictors[column] = np.log(cand_train_predictors[column])
#     elif coef[0][1] > 0.0 and coef[0][1] <= .5:
#         print('square root')
#         cand_train_predictors[column] = np.sqrt(cand_train_predictors[column])
#     elif coef[0][1] > .5 and coef[0][1] <= 1.:
#         print('no transform')
#
#     if np.std(cand_train_predictors[column]) == 0:
#         cand_train_predictors = cand_train_predictors.drop(column, axis=1)
#
#     # cand_train_predictors.fillna(cand_train_predictors.mean(), inplace=True)
#     # try:
#     #     sns.kdeplot(cand_train_predictors[column])
#     #     plt.show()
#     # except:
#     #     print(np.mean(cand_train_predictors[column]))
#     #     print(np.std(cand_train_predictors[column]))

cand_train_predictors.fillna(cand_train_predictors.mean(), inplace=True)
cand_test_predictors.fillna(cand_test_predictors.mean(), inplace=True)

skewed_feats = cand_train_predictors.apply(lambda x: skew(x))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

print('Skewed features:', skewed_feats)

cand_train_predictors[skewed_feats] = np.log1p(cand_train_predictors[skewed_feats])
cand_test_predictors[skewed_feats] = np.log1p(cand_test_predictors[skewed_feats])
#
# corr_matrix = cand_train_predictors.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# cols_to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
# print('Highly correlated features(will be droped):', cols_to_drop)
#
# cand_train_predictors = cand_train_predictors.drop(cols_to_drop, axis=1)
# cand_test_predictors = cand_test_predictors.drop(cols_to_drop, axis=1)
#
# print(cand_train_predictors.shape)
# print(cand_test_predictors.shape)

train_set, test_set = cand_train_predictors.align(cand_test_predictors, join='left', axis=1)

# print(train_set.columns)
# for year in train_set.YrSold.unique():
#     print(year, '->', len(train_set[train_set.YrSold == year]))
#     y_year = y[train_set[train_set.YrSold == year].index]
#     print(len(y_year))
#
#
# def max_len_by_year(train_set):
#     lens = []
#     for year in train_set.YrSold.unique():
#         lens.append(len(train_set[train_set.YrSold == year]))
#     return max(lens)
#
#
# print(max_len_by_year(train_set))

# regr = make_pipeline(StandardScaler(),GradientBoostingRegressor(n_estimators=1000))
regr = GradientBoostingRegressor(n_estimators=1000)
score = rmsle_cv(regr, train_set, y)
print(score)
print(np.mean(score))
# regr.fit(train_set, y)
# print(regr.score(train_set, y))
# y_pred = regr.predict(train_set)
# print(np.sqrt(mean_squared_error(y, y_pred)))
