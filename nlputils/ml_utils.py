from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range
import numpy as np


def xval(Xlist, K=10):
    """
    Input:
        - Xlist: list of ids [tip: maybe use np.random.permutation(len(Xlist)) before]
        - K: number of folds
    Generator yields:
        - training and test sets with ids
    """
    for k in range(K):
        train = [x for i, x in enumerate(Xlist) if i % K != k]
        test = [x for i, x in enumerate(Xlist) if i % K == k]
        yield train, test


def balanced_xval(Xdict, K=10, random_seed=None):
    """
    Input:
        - Xdict: dictionary with category : list of doc_ids
        - K: number of folds
    Generator yields:
        - training and test sets that are balanced with respect to
          the number of docs from each category (i.e. ratio stays the same)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    Xlist = []
    for cat in Xdict:
        Xlist.extend(list(np.random.permutation(sorted(Xdict[cat]))))
    for k in range(K):
        train = [x for i, x in enumerate(Xlist) if i % K != k]
        test = [x for i, x in enumerate(Xlist) if i % K == k]
        yield train, test
