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
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor

import warnings
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

to_str = ['YearBuilt', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'LotFrontage']
# to_few = ['Street','Utilities','LandSlope','Condition2']

for column in train_data.columns:
    if column in ['Id', 'SalePrice']:
        continue
    df = pd.DataFrame(columns=[column, 'SalePrice'])
    df['SalePrice'] = train_data.SalePrice

    if train_data[column].dtype != 'object':
        train_data[column] = train_data[column].fillna(train_data[column].mean())
        test_data[column] = test_data[column].fillna(test_data[column].mean())

    if column in to_str:
        if train_data[column].dtype == 'float64':
            train_data[column] = train_data[column].astype('int')
            test_data[column] = test_data[column].astype('int')
        train_data[column] = train_data[column].astype('object')
        test_data[column] = test_data[column].astype('object')

    if train_data[column].dtype == 'int64':
        train_data[column] = train_data[column].astype('object')
        test_data[column] = test_data[column].astype('object')

    if train_data[column].dtype == 'object':
        train_data[column] = train_data[column].fillna('NotAvailable')
        test_data[column] = test_data[column].fillna('NotAvailable')

        lbl = LabelEncoder()
        val = train_data[column].tolist()
        val.extend(test_data[column].tolist())
        lbl.fit(val)

        train_data[column] = lbl.transform(train_data[column])
        test_data[column] = lbl.transform(test_data[column])

y = np.log1p(train_data.SalePrice)
cand_train_predictors = train_data.drop(['Id', 'SalePrice'], axis=1)
cand_test_predictors = test_data.drop(['Id'], axis=1)

corr_matrix = cand_train_predictors.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
cols_to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print('Highly correlated features(will be droped):', cols_to_drop)

cand_train_predictors = cand_train_predictors.drop(cols_to_drop, axis=1)
cand_test_predictors = cand_test_predictors.drop(cols_to_drop, axis=1)

skewed_feats = cand_train_predictors.apply(lambda x: skew(x))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

print('Skewed features:', skewed_feats)

cand_train_predictors[skewed_feats] = np.log1p(cand_train_predictors[skewed_feats])
cand_test_predictors[skewed_feats] = np.log1p(cand_test_predictors[skewed_feats])

train_set, test_set = cand_train_predictors.align(cand_test_predictors, join='left', axis=1)

regr = make_pipeline(StandardScaler(),
                     BaggingRegressor(GradientBoostingRegressor(n_estimators=1000, random_state=1), random_state=1))
score = rmsle_cv(regr, train_set, y)
print(score)
print(np.mean(score))

model = GridSearchCV(regr, param_grid={}, cv=5)
model.fit(train_set, y)
print(model.score(train_set, y))
y_pred = model.predict(train_set)
print(np.sqrt(mean_squared_error(y, y_pred)))

