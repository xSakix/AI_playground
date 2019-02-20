import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, TheilSenRegressor, SGDRegressor, LassoCV
from sklearn.neural_network import MLPRegressor


def geometric_median(points, method='auto', options={}):
    """
    Calculates the geometric median of an array of points.

    method specifies which algorithm to use:
        * 'auto' -- uses a heuristic to pick an algorithm
        * 'minimize' -- scipy.optimize the sum of distances
        * 'weiszfeld' -- Weiszfeld's algorithm
    """

    points = np.asarray(points)

    if len(points.shape) == 1:
        # geometric_median((0, 0)) has too much potential for error.
        # Did the user intend a single 2D point or two scalars?
        # Use np.median if you meant the latter.
        raise ValueError("Expected 2D array")

    if method == 'auto':
        if points.shape[1] > 2:
            # weiszfeld tends to converge faster in higher dimensions
            method = 'weiszfeld'
        else:
            method = 'minimize'

    return _methods[method](points, options)


def minimize_method(points, options={}):
    """
    Geometric median as a convex optimization problem.
    """

    # objective function
    def aggregate_distance(x):
        return cdist([x], points).sum()

    # initial guess: centroid
    centroid = points.mean(axis=0)

    optimize_result = minimize(aggregate_distance, centroid, method='COBYLA')

    return optimize_result.x


def weiszfeld_method(points, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """

    default_options = {'maxiter': 1000, 'tol': 1e-7}
    default_options.update(options)
    options = default_options

    def distance_func(x):
        return cdist([x], points)

    # initial guess: centroid
    guess = points.mean(axis=0)

    iters = 0

    while iters < options['maxiter']:
        distances = distance_func(guess).T

        # catch divide by zero
        # TODO: Wikipedia cites how to deal with distance 0
        distances = np.where(distances == 0, 1, distances)

        guess_next = (points / distances).sum(axis=0) / (1. / distances).sum(axis=0)

        guess_movement = np.sqrt(((guess - guess_next) ** 2).sum())

        guess = guess_next

        if guess_movement <= options['tol']:
            break

        iters += 1

    return guess


def ridge_regressor(points, options={}):
    regr1 = RidgeCV(alphas=[0.1, 0.5, 1., 2., 5., 10.])
    x1 = points[0:-1, 0].reshape(-1, 1)
    y1 = points[1:, 0]
    regr1.fit(x1, y1)

    regr2 = RidgeCV(alphas=[0.1, 0.5, 1., 2., 5., 10.])
    x2 = points[0:-1, 1].reshape(-1, 1)
    y2 = points[1:, 1]
    regr2.fit(x2, y2)

    # print(regr1.score(x1, y1), ' , ', regr2.score(x2, y2))

    return [regr1.predict(points[-1, 0].reshape((-1, 1)))[0], regr2.predict(points[-1, 1].reshape((-1, 1)))[0]]


def lasso_regressor(points, options={}):
    result = []
    for i in range(points.shape[1]):
        regr = Lasso(alpha=0.)
        x1 = points[0:-1, i].reshape(-1, 1)
        y1 = points[1:, i]
        regr.fit(x1, y1)
        result.append(regr.predict(points[-1, i].reshape((-1, 1)))[0])

    return result


def mlp_regressor(points, options={}):
    result = []
    for i in range(points.shape[1]):
        regr = MLPRegressor(activation='relu')
        x1 = points[0:-1, i].reshape(-1, 1)
        y1 = points[1:, i]
        regr.fit(x1, y1)
        result.append(regr.predict(points[-1, i].reshape((-1, 1)))[0])

    return result

def sgd_regressor(points, options={}):
    regr1 = SGDRegressor(penalty='l1')
    x1 = points[0:-1, 0].reshape(-1, 1)
    y1 = points[1:, 0]
    regr1.fit(x1, y1)

    regr2 = SGDRegressor(penalty='l1')
    x2 = points[0:-1, 1].reshape(-1, 1)
    y2 = points[1:, 1]
    regr2.fit(x2, y2)

    # print(regr1.score(x1, y1), ' , ', regr2.score(x2, y2))

    return [regr1.predict(points[-1, 0].reshape((-1, 1)))[0], regr2.predict(points[-1, 1].reshape((-1, 1)))[0]]

def rf_regressor(points, options={}):
    result = []
    for i in range(points.shape[1]):
        regr = RandomForestRegressor()
        x1 = points[0:-1, i].reshape(-1, 1)
        y1 = points[1:, i]
        regr.fit(x1, y1)
        result.append(regr.predict(points[-1, i].reshape((-1, 1)))[0])

    return result


_methods = {
    'minimize': minimize_method,
    'weiszfeld': weiszfeld_method,
    'ridge_regressor': ridge_regressor,
    'lasso_regressor': lasso_regressor,
    'sgd_regressor': sgd_regressor,
    'mlp_regressor': mlp_regressor,
    'rf_regressor' : rf_regressor
}
