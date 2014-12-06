from __future__ import division
from collections import defaultdict
from math import sqrt, log
import numpy as np
from sys import float_info as fi
from nlputils.misc import make_featmat

def _polynomial_sim(x,y, p=2):
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
    return float(sum([x[word]*y[word] for word in s]))**p

def _sigmoidal_sim(x,y):
    """
    sigmoidal kernel
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in both x and y (for all others the product is 0 anyways)
    s = set(x.keys()) & set(y.keys())
    # nothing in common?
    if not s:
        return 0.
    return np.tanh(sum([x[word]*y[word] for word in s]))

def _histint_sim(x,y):
    """
    histogram intersection kernel
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in both x and y (for all others min is 0 anyways)
    s = set(x.keys()) & set(y.keys())
    # nothing in common?
    if not s:
        return 0.
    return float(sum([min(x[word],y[word]) for word in s]))

def _gaussian_sim(x,y, gamma=.25):
    """
    gaussian rbf kernel with width 1/gamma
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) & set(y.keys())
    sx = set(x.keys()).difference(s)
    sy = set(y.keys()).difference(s)
    return np.exp(-gamma*float(sum([(x[word]-y[word])**2 for word in s])+sum([(x[word])**2 for word in sx])+sum([(y[word])**2 for word in sy])))

def _minkowski_sim(x,y, p=3):
    """
    minkowski distance: p=1: manhattan, p=2: squared euclidean, etc. returned with - as a pseudo similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) & set(y.keys())
    sx = set(x.keys()).difference(s)
    sy = set(y.keys()).difference(s)
    return 1./(fi.epsilon + float(sum([np.abs(x[word]-y[word])**p for word in s])+sum([x[word]**p for word in sx])+sum([y[word]**p for word in sy])))

def _canberra_sim(x,y, p=1):
    """
    canberra distance p=1 / chi^2 with p=2; returned with - as a pseudo similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) | set(y.keys())
    x = defaultdict(float,x)
    y = defaultdict(float,y)
    return 1./(fi.epsilon + float(sum([np.abs(x[word]-y[word])**p/float(x[word]+y[word]) for word in s])))

def _chebyshev_sim(x,y):
    """
    1 - chebyshev distance, as a pseudo similarity, only works if counts are normalized to be <= 1
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) | set(y.keys())
    x = defaultdict(float,x)
    y = defaultdict(float,y)
    return 1.- float(max([np.abs(x[word]-y[word]) for word in s]))

def _hellinger_sim(x,y):
    """
    hellinger distance, returned with - as a pseudo similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) & set(y.keys())
    sx = set(x.keys()).difference(s)
    sy = set(y.keys()).difference(s)
    return 1./(fi.epsilon + (sum([(sqrt(float(x[word]))-sqrt(float(y[word])))**2 for word in s])+float(sum([x[word] for word in sx]))+float(sum([y[word] for word in sy]))))

def _jenshan_sim(x,y):
    """
    jensen-shannon distance, returned with - as a pseudo similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    # get the words that occur in either x or y
    s = set(x.keys()) | set(y.keys())
    x = defaultdict(float,x)
    y = defaultdict(float,y)
    def H(xcount, ycount):
        h = 0.
        if xcount:
            h += xcount*log(2.*xcount/(xcount+ycount))
        if ycount:
            h+= ycount*log(2.*ycount/(xcount+ycount))
        return h
    return 1./(fi.epsilon + float(sum([H(float(x[word]),float(y[word])) for word in s])))

def _simpson_sim(x,y):
    """
    simpson similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    return _histint_sim(x,y)/min(_histint_sim(x,x),_histint_sim(y,y))

def _braun_sim(x,y):
    """
    braun-blanquet similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    return _histint_sim(x,y)/max(_histint_sim(x,x),_histint_sim(y,y))

def _kulczynski_sim(x,y):
    """
    kulczynski (2) similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    a = _histint_sim(x,y)
    return 0.5*(a/_histint_sim(x,x)+a/_histint_sim(y,y))

def _jaccard_sim(x,y):
    """
    jaccard similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    a = _histint_sim(x,y)
    return a/(_histint_sim(x,x)+_histint_sim(y,y)-a)

def _dice_sim(x,y):
    """
    czekanowski, sorensen-dice similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    return 2.*_histint_sim(x,y)/(_histint_sim(x,x)+_histint_sim(y,y))

def _otsuka_sim(x,y):
    """
    otsuka, ochiai similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    return _histint_sim(x,y)/(sqrt(_histint_sim(x,x))*sqrt(_histint_sim(y,y)))

def _sokal_sim(x,y):
    """
    sokal-sneath, anderberg similarity
    compute the similarity between the dicts x and y with terms + their counts
    """
    a = _histint_sim(x,y)
    return a/(2.*_histint_sim(x,x)+2.*_histint_sim(y,y)-3.*a)

def compute_sim(x, y, sim='linear'):
    """
    compute the similarity between x and y
    Input:
        x, y: dicts representing 2 documents (all normalization, weighting, etc has to be done before!)
        sim: the kind of similarity to be computed:
                - kernels: ['linear', 'polynomial', 'sigmoidal', 'histint', 'gaussian']
                - (neg) distance: ['manhattan', 'sqeucl', 'minkowski', 'canberra', 'chisq', 'chebyshev', 'hellinger', 'jenshan']
                - similarity coef: ['simpson', 'braun', 'kulczynski', 'jaccard', 'dice', 'otsuka', 'sokal']
    Returns:
        a single number representing the similarity between x and y
    """
    if sim == 'linear':
        return _polynomial_sim(x,y,1)
    elif sim == 'polynomial':
        return _polynomial_sim(x,y)
    elif sim == 'sigmoidal':
        return _sigmoidal_sim(x,y)
    elif sim == 'histint':
        return _histint_sim(x,y)
    elif sim == 'gaussian':
        return _gaussian_sim(x,y)
    elif sim == 'minkowski':
        return _minkowski_sim(x,y)
    elif sim == 'sqeucl':
        return _minkowski_sim(x,y,2)
    elif sim == 'manhattan':
        return _minkowski_sim(x,y,1)
    elif sim == 'canberra':
        return _canberra_sim(x,y)
    elif sim == 'chisq':
        return _canberra_sim(x,y,2)
    elif sim == 'chebyshev':
        return _chebyshev_sim(x,y)
    elif sim == 'hellinger':
        return _hellinger_sim(x,y)
    elif sim == 'jenshan':
        return _jenshan_sim(x,y)
    elif sim == 'simpson':
        return _simpson_sim(x,y)
    elif sim == 'braun':
        return _braun_sim(x,y)
    elif sim == 'kulczynski':
        return _kulczynski_sim(x,y)
    elif sim == 'jaccard':
        return _jaccard_sim(x,y)
    elif sim == 'dice':
        return _dice_sim(x,y)
    elif sim == 'otsuka':
        return _otsuka_sim(x,y)
    elif sim == 'sokal':
        return _sokal_sim(x,y)
    else:
        print "ERROR: sim not known!!"
        return None

def compute_K(doc_ids, docfeats, sim='linear', normalize=False):
    """
    Imput:
        doc_ids: list of document ids (keys of docfeats)
        docfeats: dict with doc_id:{feat:count}
        sim: type of similarity
        normalize: if the matrix should be normalized to have 1 on the diagonals and 1 <= s <= 0 on off diagonal
                   by dividing k(x,y) by the mean of k(x,x) and k(y,y)
    Returns:
        symmetric similarity matrix of size len(doc_ids)xlen(doc_ids)
    """
    # if linear similarity, we're quicker with a dot product
    if sim == 'linear':
        featmat, _ = make_featmat(docfeats, doc_ids, featurenames=[])
        S = featmat.dot(featmat.T).toarray()
    else:
        # compute general similarity matrix
        N = len(doc_ids)
        S = np.zeros((N,N))
        for i, docid in enumerate(doc_ids):
            for j in range(i+1):
                similarity = compute_sim(docfeats[docid],docfeats[doc_ids[j]],sim)
                S[i,j], S[j,i] = similarity, similarity
    if normalize:
        diag_elem = np.array([np.diag(S)])
        norm = (np.tile(diag_elem,(N,1)) + np.tile(diag_elem.T,(1,N)))/2.
        S /= norm
    return S

def compute_K_map(train_ids, test_ids, docfeats, sim='linear', train_idx=[], normalize=False):
    """
    Imput:
        train_ids, test_ids: list of document ids (keys of docfeats)
        docfeats: dict with doc_id:feat
        sim: type of similarity
        train_idx: list of indexes for train_ids, e.g. for svms we don't need all training examples
    Returns:
        kernel map of size len(test_ids)xlen(train_ids) with similarities of the test to the training docs
    """
    # if linear similarity, we're quicker with a dot product
    if sim == 'linear':
        featmat_train, featurenames = make_featmat(docfeats, train_ids)
        featmat_test, featurenames = make_featmat(docfeats, test_ids, featurenames)
        K_map = featmat_test.dot(featmat_train.T).toarray()
    else:
        # compute similarity of all test examples to all training examples
        n_tr = len(train_ids)
        n_ts = len(test_ids)
        if not len(train_idx):
            train_idx = range(n_tr)
        K_map = np.array([[compute_sim(docfeats[did_ts],docfeats[train_ids[j]],sim) for j in train_idx] for did_ts in test_ids])
    if normalize:
        train_sim = np.array([[compute_sim(docfeats[did_tr],docfeats[did_tr],sim) for did_tr in train_ids]])
        test_sim = np.array([[compute_sim(docfeats[did_ts],docfeats[did_ts],sim) for did_ts in test_ids]]).T
        norm = (np.tile(train_sim,(n_ts,1)) + np.tile(test_sim,(1,n_tr)))/2.
        K_map /= norm
    return K_map


def get_k_most_similar(K, row_ids, col_ids, did, k=10):
    """
    given the kernel matrix and corresponding doc_ids, get the k most similar documents corresponding to the doc did
    
    Input:
        K: kernel matrix or map with similarities (the higher the more similar)
        row_ids: list of ids in the order of the rows in K
        col_ids: list of ids in the order of the cols in K (same as row_ids in the case of the kernel matrix, not map)
        did: a single doc_ids corresponding to one row in K and to which similar documents should be found
        k: how many similar documents should be returned
    Returns:
        simdocids: [(docid, simscore)] a list of k docids and their respective similarity to the target did
    """
    try:
        dididx = row_ids.index(did)
    except ValueError:
        print "did must have a designated row in K"
        return []
    # get k indexes for K
    k_idx = np.flipud(np.argsort(K[dididx,:]))[:k+1]
    return [(col_ids[i],K[dididx,i]) for i in k_idx if not col_ids[i]==did][:k]
