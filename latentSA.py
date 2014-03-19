from __future__ import division
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from similarity import compute_K
from misc import vec2dict, invert_dict2, sim_ev, select_copy

def compute_LSA(docfeats, doc_ids=[], n_ev=3000):
    """
    Input:
        docfeats: a dict with doc_id:features where features is a dict of term:counts 
        doc_ids: which docs in docfeats should be used to compute the weights
        n_ev: how many eigenvectors should be kept
    Returns:
        V: the first n_ev eigenvectors (in columns)
        terms: a list of terms corresponding to the indexes of the rows of V
    """
    # delete all documents that shouldn't be considered (e.g. in xval)
    if doc_ids:
        docfeats = select_copy(docfeats, doc_ids)
    # invert the docfeat dict to have terms as keys
    termdict = invert_dict2(docfeats)
    # get a list of terms
    terms = sorted(termdict.keys())
    # get term-term interaction matrix
    T = compute_K(terms, termdict)
    # compute eigenvalues of T
    print "computing eigenvalues of term matrix"
    print str(datetime.now())
    D, V = sim_ev(T)
    n_ev = min(n_ev,len(D))
    # plot for checking
    plt.figure()
    plt.plot(range(1,len(D)+1),D)
    plt.xlabel('Index of eigenvalue')
    plt.ylabel('Magnitude of eigenvalue')
    plt.title('LSA - eigenvalue spectrum')
    plt.show()
    # take only first n_ev eigenvectors
    print "Taken %i eigenvectors from a total of %i and they explain %.2f of the variance"%(n_ev,len(D),(np.sum(D[:n_ev])/np.sum(D)))
    return V[:,:n_ev], terms

def apply_LSA(docfeats, V, terms, doc_ids=[]):
    """
    Input:
        docfeats: a dict with doc_id:features where features is a dict of term:counts
        doc_ids: which docs in docfeats should be used to compute the weights
        V: the eigenvectors (in columns) from LSA
        terms: a list of terms corresponding to the indexes of the rows of V
    Returns:
        lsa_features: a dict with doc_id:{i_evec:vectorval} (projection of terms/counts onto V)
    """
    if not doc_ids:
        doc_ids = docfeats.keys()
    # make feature vectors for each question
    termidx = {term:i for i, term in enumerate(terms)}
    features = {}
    for did in doc_ids:
        idx = []
        counts = []
        for term in docfeats[did]:
            #idx.append(terms.index(term))
            idx.append(termid[term])
            counts.append(docfeats[did][term])
        # weight by the counts
        features[did] = vec2dict(np.dot(counts,V[idx,:]))
    return features
