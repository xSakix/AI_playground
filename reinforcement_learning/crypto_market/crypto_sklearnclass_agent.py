import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, \
    GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.classification import classification_report, accuracy_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier


def sgd_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...SGDClassifier')
    clf = SGDClassifier()

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))

    joblib.dump(clf, dir_models + ticket + '_sgd_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def qda_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...QuadraticDiscriminantAnalysis')
    clf = QuadraticDiscriminantAnalysis()

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))

    joblib.dump(clf, dir_models + ticket + '_qda_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def adaboost_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...Ada')
    clf = AdaBoostClassifier()

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_adaboost_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def mlp_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...MLP')
    clf = MLPClassifier(early_stopping=True)

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_mlp_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def random_forrest(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...RandomForest')
    clf = RandomForestClassifier(verbose=False, warm_start=True)
    print('training...')
    clf.fit(x, y)
    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))
    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_random_forrest_' + str(id) + '.pkl')
    return clf.score(x_test, y_test)


def decision_tree(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...decision tree')
    clf = DecisionTreeClassifier()
    print('training...')
    clf.fit(x, y)
    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))
    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_decission_tree_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def voting_random_forrest(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...voting RandomForest')
    estimators = [(str(idd), RandomForestClassifier()) for idd in range(100)]
    clf = VotingClassifier(estimators=estimators)
    print('training...')
    clf.fit(x, y)
    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))
    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_voting_rand_forest_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def voting_decision_tree(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...Voting decision tree')
    estimators = [(str(idd), DecisionTreeClassifier()) for idd in range(100)]
    clf = VotingClassifier(estimators=estimators, voting='soft')
    print('training...')
    clf.fit(x, y)
    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))
    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_voting_dectree_' + str(id) + '.pkl')

    return accuracy_score(y_test, predicted)


def gaussnb_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...GaussianNB')
    clf = GaussianNB()

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    score = accuracy_score(y_test, predicted)
    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_gausianNB_' + str(id) + '.pkl')

    return score


def kneighbors_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...KNeighbors')
    clf = KNeighborsClassifier(3)

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_kneighbors_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def bernoulli_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...BernoulliNB')
    clf = BernoulliNB(binarize=True)

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_bernoulli_' + str(id) + '.pkl')
    return clf.score(x_test, y_test)


def bayes_gauss_classifier(dir_models, ticket, x, x_test, y, y_test):
    # GaussianMixture
    print('getting model...BayesianGaussianMixture')
    clf = BayesianGaussianMixture(n_components=3)

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_bayesian_gaussian_mixture_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def gaussmix_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...GaussianMixture')
    clf = GaussianMixture(n_components=3)

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_gaussian_mixture_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def linearsvc_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...linear svc')
    clf = LinearSVC()

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_linear_svc_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


def svc_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...SVC')
    clf = SVC()

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_svc_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)

def nusvc_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...NuSVC')
    clf = NuSVC(nu=0.8)

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_nusvc_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)

def bagging_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...Bagging')
    clf = BaggingClassifier()

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_bag_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)

def gradient_classifier(dir_models, ticket, x, x_test, y, y_test):
    print('getting model...GBC')
    clf = GradientBoostingClassifier(n_estimators=1000)

    print('training...')
    clf.fit(x, y)

    print('predicting...')
    predicted = clf.predict(x_test)
    print(classification_report(y_test, predicted))

    id = len(os.listdir(dir_models))
    joblib.dump(clf, dir_models + ticket + '_gbc_' + str(id) + '.pkl')

    return clf.score(x_test, y_test)


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

    result = []

    # result.append(decision_tree(dir_models, ticket, x, x_test, y, y_test))
    result.append(random_forrest(dir_models, ticket, x, x_test, y, y_test))
    # result.append(voting_random_forrest(dir_models, ticket, x, x_test, y, y_test))
    # result.append(voting_decision_tree(dir_models, ticket, x, x_test, y, y_test))
    # result.append(gaussnb_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(kneighbors_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(mlp_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(adaboost_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(qda_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(sgd_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(bernoulli_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(bayes_gauss_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(gaussmix_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(linearsvc_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(svc_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(nusvc_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(bagging_classifier(dir_models, ticket, x, x_test, y, y_test))
    # result.append(gradient_classifier(dir_models, ticket, x, x_test, y, y_test))

    return result


if __name__ == '__main__':
    dir_data = 'data_ga_periodic'
    dir_models = 'models_ga_periodic'
    report = run_learning(dir_data + '/', dir_models + '/')
    print(report)
