from __future__ import unicode_literals, division, print_function, absolute_import
from collections import defaultdict
from sys import float_info as fi
from math import sqrt, log
import numpy as np


def _polynomial_sim(x, y, p=2):
    """
    polynomial/linear kernel
    compute the similarity between the dicts x and y with terms + their counts
    p: >1 for polynomial kernel (default = 2)
    """
    # get the words that occur in both x and y (for all others the product is 0 anyways)
    s = set(x.keys()) & set(y.keys())
    # nothing in common?
    if not s:
        return 0.
    return float(sum([x[word] * y[word] for word in s]))**p


def _sigmoidal_sim(x, y):
    """
    sigmoidal kernel
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in both x and y (for all others the product is 0 anyways)
    s = set(x.keys()) & set(y.keys())
    # nothing in common?
    if not s:
        return 0.
    return np.tanh(sum([x[word] * y[word] for word in s]))


def _histint_sim(x, y):
    """
    histogram intersection kernel
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in both x and y (for all others min is 0 anyways)
    s = set(x.keys()) & set(y.keys())
    # nothing in common?
    if not s:
        return 0.
    return float(sum([min(x[word], y[word]) for word in s]))


def _gaussian_sim(x, y, gamma=.25):
    """
    gaussian rbf kernel with width 1/gamma
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) & set(y.keys())
    sx = set(x.keys()).difference(s)
    sy = set(y.keys()).difference(s)
    return np.exp(-gamma * float(sum([(x[word] - y[word])**2 for word in s]) + sum([(x[word])**2 for word in sx]) + sum([(y[word])**2 for word in sy])))


def _minkowski_sim(x, y, p=3):
    """
    minkowski distance: p=1: manhattan, p=2: squared euclidean, etc. returned with - as a pseudo similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) & set(y.keys())
    sx = set(x.keys()).difference(s)
    sy = set(y.keys()).difference(s)
    return 1. / (fi.epsilon + float(sum([np.abs(x[word] - y[word])**p for word in s]) + sum([x[word]**p for word in sx]) + sum([y[word]**p for word in sy])))


def _canberra_sim(x, y, p=1):
    """
    canberra distance p=1 / chi^2 with p=2; returned with - as a pseudo similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) | set(y.keys())
    x = defaultdict(float, x)
    y = defaultdict(float, y)
    return 1. / (fi.epsilon + float(sum([np.abs(x[word] - y[word])**p / float(x[word] + y[word]) for word in s])))


def _chebyshev_sim(x, y):
    """
    1 - chebyshev distance, as a pseudo similarity, only works if counts are normalized to be <= 1
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) | set(y.keys())
    x = defaultdict(float, x)
    y = defaultdict(float, y)
    return 1. - float(max([np.abs(x[word] - y[word]) for word in s]))


def _hellinger_sim(x, y):
    """
    hellinger distance, returned with - as a pseudo similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) & set(y.keys())
    sx = set(x.keys()).difference(s)
    sy = set(y.keys()).difference(s)
    return 1. / (fi.epsilon + (sum([(sqrt(float(x[word])) - sqrt(float(y[word])))**2 for word in s]) + float(sum([x[word] for word in sx])) + float(sum([y[word] for word in sy]))))


def _jenshan_sim(x, y):
    """
    jensen-shannon distance, returned with - as a pseudo similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) | set(y.keys())
    x = defaultdict(float, x)
    y = defaultdict(float, y)

    def H(xcount, ycount):
        h = 0.
        if xcount:
            h += xcount * log(2. * xcount / (xcount + ycount))
        if ycount:
            h += ycount * log(2. * ycount / (xcount + ycount))
        return h
    return 1. / (fi.epsilon + float(sum([H(float(x[word]), float(y[word])) for word in s])))


def _simpson_sim(x, y):
    """
    simpson similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    return _histint_sim(x, y) / min(_histint_sim(x, x), _histint_sim(y, y))


def _braun_sim(x, y):
    """
    braun-blanquet similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    return _histint_sim(x, y) / max(_histint_sim(x, x), _histint_sim(y, y))


def _kulczynski_sim(x, y):
    """
    kulczynski (2) similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    a = _histint_sim(x, y)
    return 0.5 * (a / _histint_sim(x, x) + a / _histint_sim(y, y))


def _jaccard_sim(x, y):
    """
    jaccard similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    a = _histint_sim(x, y)
    return a / (_histint_sim(x, x) + _histint_sim(y, y) - a)


def _dice_sim(x, y):
    """
    czekanowski, sorensen-dice similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    return 2. * _histint_sim(x, y) / (_histint_sim(x, x) + _histint_sim(y, y))


def _otsuka_sim(x, y):
    """
    otsuka, ochiai similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    return _histint_sim(x, y) / (sqrt(_histint_sim(x, x)) * sqrt(_histint_sim(y, y)))


def _sokal_sim(x, y):
    """
    sokal-sneath, anderberg similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    a = _histint_sim(x, y)
    return a / (2. * _histint_sim(x, x) + 2. * _histint_sim(y, y) - 3. * a)


def compute_sim(x, y, sim='linear'):
    """
    compute the similarity between x and y
    Input:
        - x, y: dicts representing 2 documents (all normalization, weighting, etc has to be done before!)
        - sim: the kind of similarity to be computed:
                -- kernels: ['linear', 'polynomial', 'sigmoidal', 'histint', 'gaussian']
                -- (neg) distance: ['manhattan', 'sqeucl', 'minkowski', 'canberra', 'chisq', 'chebyshev', 'hellinger', 'jenshan']
                -- similarity coef: ['simpson', 'braun', 'kulczynski', 'jaccard', 'dice', 'otsuka', 'sokal']
    Returns:
        - a single number representing the similarity between x and y
    """
    if sim == 'linear':
        return _polynomial_sim(x, y, 1)
    elif sim == 'polynomial':
        return _polynomial_sim(x, y)
    elif sim == 'sigmoidal':
        return _sigmoidal_sim(x, y)
    elif sim == 'histint':
        return _histint_sim(x, y)
    elif sim == 'gaussian':
        return _gaussian_sim(x, y)
    elif sim == 'minkowski':
        return _minkowski_sim(x, y)
    elif sim == 'sqeucl':
        return _minkowski_sim(x, y, 2)
    elif sim == 'manhattan':
        return _minkowski_sim(x, y, 1)
    elif sim == 'canberra':
        return _canberra_sim(x, y)
    elif sim == 'chisq':
        return _canberra_sim(x, y, 2)
    elif sim == 'chebyshev':
        return _chebyshev_sim(x, y)
    elif sim == 'hellinger':
        return _hellinger_sim(x, y)
    elif sim == 'jenshan':
        return _jenshan_sim(x, y)
    elif sim == 'simpson':
        return _simpson_sim(x, y)
    elif sim == 'braun':
        return _braun_sim(x, y)
    elif sim == 'kulczynski':
        return _kulczynski_sim(x, y)
    elif sim == 'jaccard':
        return _jaccard_sim(x, y)
    elif sim == 'dice':
        return _dice_sim(x, y)
    elif sim == 'otsuka':
        return _otsuka_sim(x, y)
    elif sim == 'sokal':
        return _sokal_sim(x, y)
    else:
        print("ERROR: sim not known!!")
        return None
