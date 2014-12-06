from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from copy import deepcopy

def make_featmat(docfeats, doc_ids, featurenames=[]):
    """
    Transform a dictionary with features into a sparse matrix for sklearn algorithms
    Input:
        - docfeats: a dictionary with {docid:{word:count}}
        - doc_ids: the subset of the docfeats (keys to the dict) that should be regarded, 
                   defines rows of the feature matrix
        - featurenames: a list of words that define the columns of the feature matrix
                --> when the doc_ids are the training samples, this can be left empty
                    and the words in the training document will be used; for testing give the returned names
    Returns:
        - featmat: a sparse matrix with doc_ids x featurenames
        - featurenames: the list of words defining the columns of the featmat

    Example usage:
        features_train, featurenames = make_featmat(docfeats, trainids)
        features_test, featurenames = make_featmat(docfeats, testids, featurenames)
    """
    if not featurenames:
        featurenames = sorted(invert_dict1(select_copy(docfeats,doc_ids)).keys())
    fnamedict = {feat:i for i, feat in enumerate(featurenames)}
    featmat = dok_matrix((len(doc_ids),len(featurenames)), dtype=float)
    for i, did in enumerate(doc_ids):
        for word in docfeats[did]:
            try:
                featmat[i,fnamedict[word]] = docfeats[did][word]
            except KeyError:
                pass
    featmat = csr_matrix(featmat)
    return featmat, featurenames
    

def xval(Xlist, K=10):
    """
    Input:
        - Xlist: list of ids [tip: maybe use np.random.permutation(len(Xlist)) before]
        - K: number of folds
    Generator yields:
        - training and test sets with ids
    """
    for k in xrange(K):
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
        Xlist.extend(list(np.random.permutation(Xdict[cat])))
    for k in xrange(K):
        train = [x for i, x in enumerate(Xlist) if i % K != k]
        test = [x for i, x in enumerate(Xlist) if i % K == k]
        yield train, test

def sim_ev(K):
    """
    compute the eigenvalues and -vectors of a similarity matrix
    Input:
        K: a symmetric nxn similarity matrix (for dissimilarity matrices, first multiply by -1/2)
    Return:
        D: a vector containing the eigenvalues of K (sorted decreasing)
        V: the eigenvectors corresponding to D 
    """
    n, m = K.shape
    H = np.eye(n) - np.tile(1./n,(n,n))
    B = np.dot(np.dot(H,K),H)
    D, V = np.linalg.eigh((B+B.T)/2.) # guard against spurious complex e-vals from roundoff
    D, V = D.real, V.real
    I = np.flipud(np.argsort(D)) # sort descending
    D, V = D[I], V[:,I]
    return D, V


