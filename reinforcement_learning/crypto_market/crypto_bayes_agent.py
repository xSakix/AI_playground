import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.classification import classification_report
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def run_learning(dir_data='data_btc_eur/', dir_models=None):
    if dir_models is None:
        models = [d for d in os.listdir('.') if d.startswith('models_btc_eur')]
        dir_models = 'models_btc_eur_' + str(len(models) + 1) + '/'

    if not os.path.isdir(dir_models):
        print('creating dir...' + dir_models)
        os.makedirs(dir_models)

    ticket = 'BTC_EUR'

    print('loading data...')
    x = np.load(dir_data + 'x.npy')
    if len(x.shape) == 1:
        transform = lambda x: x.replace('array', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        x = np.array([np.fromstring(transform(xx), sep=',') for xx in x])

    print(x.shape)
    x = np.nan_to_num(x)

    y_orig = np.load(dir_data + 'y.npy')

    print('reshaping data...')
    if len(x.shape) > 2:
        if x.shape[2] == 1:
            x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        else:
            x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        print(x.shape)

    if x.shape[1] == 7:
        x = x[:, 2:7]

    print(x.shape)

    print('min:', np.min(x))
    print('max:', np.max(x))

    labels = ['ror', 'bench', 'lowerb', 'mean', 'median', 'higherb']

    if len(y_orig.shape) > 1:
        y = y_orig.reshape(y_orig.shape[0] * y_orig.shape[1])
    else:
        y = y_orig
    print('y_orig:', y.shape)

    unique, counts = np.unique(y_orig, return_counts=True)
    print(dict(zip(unique, counts)))

    print('spliting data...')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

    # -------------------classifiers

    # print('getting model...BernoulliNB')
    # clf = BernoulliNB(binarize=True)
    #
    # print('training...')
    # clf.fit(x, y)
    #
    # print('predicting...')
    # predicted = clf.predict(x_test)
    # print(classification_report(y_test, predicted))
    #
    # id = len(os.listdir(dir_models))
    # joblib.dump(clf, dir_models + ticket + '_bernoulli_' + str(id) + '.pkl')

    # print('getting model...GaussianNB')
    # clf = GaussianNB()
    #
    # print('training...')
    # clf.fit(x, y)
    #
    # print('predicting...')
    # predicted = clf.predict(x_test)
    # print(classification_report(y_test, predicted))
    #
    # id = len(os.listdir(dir_models))
    # joblib.dump(clf, dir_models + ticket + '_gauss_' + str(id) + '.pkl')

    # print('getting model...Boltzmann + GaussianNB')
    # clf = Pipeline([('bern', BernoulliRBM(n_components=x.shape[1])), ('clf', GaussianNB())])
    #
    # print('training...')
    # clf.fit(x, y)
    #
    # print('predicting...')
    # predicted = clf.predict(x_test)
    # print(classification_report(y_test, predicted))
    #
    # id = len(os.listdir(dir_models))
    # joblib.dump(clf, dir_models + ticket + '_boltz_gauss_' + str(id) + '.pkl')

    # print('getting model...Boltzmann + BernoulliNB')
    # clf = Pipeline([('bern', BernoulliRBM(n_components=x.shape[1])), ('clf', BernoulliNB(binarize=True))])
    #
    # print('training...')
    # clf.fit(x, y)
    #
    # print('predicting...')
    # predicted = clf.predict(x_test)
    # print(classification_report(y_test, predicted))
    #
    # id = len(os.listdir(dir_models))
    # joblib.dump(clf, dir_models + ticket + '_boltz_bern_' + str(id) + '.pkl')
    #
    #GaussianMixture
    # print('getting model...BayesianGaussianMixture')
    # clf = BayesianGaussianMixture(n_components=3)
    #
    # print('training...')
    # clf.fit(x, y)
    #
    # print('predicting...')
    # predicted = clf.predict(x_test)
    # print(classification_report(y_test, predicted))
    #
    # id = len(os.listdir(dir_models))
    # joblib.dump(clf, dir_models + ticket + '_bayesian_gaussian_mixture_' + str(id) + '.pkl')
    #
    #
    print('getting model...GaussianMixture')
    clf = GaussianMixture(n_components=3)

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_gaussian_mixture_' + str(id) + '.pkl')

if __name__ == '__main__':
    run_learning(dir_data='data_ga_periodic/', dir_models='models_ga_periodic/')
