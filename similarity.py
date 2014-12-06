from __future__ import division
from collections import defaultdict
from math import sqrt, log
import numpy as np
from sys import float_info as fi
from nlputils.misc import make_featmat


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
