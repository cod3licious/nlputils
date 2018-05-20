from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range
import numpy as np
from .features import features2mat
from .simcoefs import compute_sim


def dist2kernel(D, perp=30, tol=10e-4, verbose=False):
    """
    Given a distance matrix, select an appropriate sigma for every datapoint such that the perplexity of
    the resulting distribution is equal to perp (see tsne paper)
    Input:
        - D: NxN matrix with distances for all N points to each other
        - perp: desired perplexity (~ # of nearest neighbors)
    Returns:
        - joint probabilities
    """
    logPerp = np.log2(perp)
    # initialize sigmas
    sigmas = np.ones(D.shape[0])
    sigmas_max = 250 * np.ones(D.shape[0])  # unrealistically high value
    sigmas_min = np.zeros(D.shape[0])
    for it in range(50):
        # compute P using the individual sigmas
        P = np.exp(-D * sigmas)
        np.fill_diagonal(P, 0.)    # set diagonal to 0
        # assume the values for each data point are in a column
        P /= np.sum(P, 0)          # normalize
        # compute the entropy of every distribution and how far we're off the desired entropy
        hdiffs = -np.sum(P * np.log2(np.maximum(P, 1e-12)), 0) - logPerp
        err = np.sum(np.abs(hdiffs))
        if verbose:
            print("iteration %i: sum of errors: %.2f" % (it, err))
        if err <= tol:
            break
        # update sigmas
        sigmas_min[hdiffs >= 0] = sigmas[hdiffs >= 0]
        sigmas_max[hdiffs <= 0] = sigmas[hdiffs <= 0]
        sigmas[hdiffs > 0] = 0.5 * (sigmas[hdiffs > 0] + sigmas_max[hdiffs > 0])
        sigmas[hdiffs < 0] = 0.5 * (sigmas[hdiffs < 0] + sigmas_min[hdiffs < 0])
    # sanity check
    if verbose:
        print(np.histogram(sigmas, 20))
    # compute P for real
    P = np.exp(-D * sigmas)
    np.fill_diagonal(P, 0.)     # set diagonal to 0
    P /= np.sum(P, 0)           # normalize
    # and from the conditional probabilities, compute the joint probabilities
    return (P + P.T) / (2. * D.shape[0])


def compute_K(docids, docfeats, sim='linear', normalize=False):
    """
    Imput:
        - docids: list of document ids (keys of docfeats)
        - docfeats: dict with doc_id:{feat:count}
        - sim: type of similarity
        - normalize: if the matrix should be normalized to have 1 on the diagonals and 1 <= s <= 0 on off diagonal
                     by dividing k(x,y) by the mean of k(x,x) and k(y,y)
    Returns:
        - symmetric similarity matrix of size len(docids)xlen(docids)
    """
    # if linear similarity or variant thereof, we're quicker with a dot product
    if sim in ('linear', 'cosine', 'angularsim', 'angulardist'):
        # transform features into matrix
        if isinstance(docfeats, dict):
            featmat, _ = features2mat(docfeats, docids, featurenames=[])
        else:
            featmat = docfeats
        # possibly normalize vectors - doesn't work yet because the featmat is sparse and numpy stupid
        # if sim in ('cosine', 'angularsim', 'angulardist'):
        #     fnorm = np.linalg.norm(featmat, axis=1)
        #     featmat /= fnorm.reshape(featmat.shape[0],1)
        # compute similarity matrix
        S = featmat.dot(featmat.T).toarray()
        # transform further
        if sim in ('angularsim', 'angulardist'):
            # make sure values are in the correct range (roundoff errors)
            S = np.minimum(S, 1)
            S = np.maximum(S, 0)
            if sim == 'angularsim':
                S = 1. - 2. * np.arccos(S) / np.pi
            else:
                # distance
                S = 2. * np.arccos(S) / np.pi
    else:
        # compute general similarity matrix
        N = len(docids)
        S = np.zeros((N, N))
        for i, did in enumerate(docids):
            for j in range(i + 1):
                similarity = compute_sim(docfeats[did], docfeats[docids[j]], sim)
                S[i, j], S[j, i] = similarity, similarity
    if normalize:
        diag_elem = np.array([np.diag(S)])
        norm = (np.tile(diag_elem, (N, 1)) + np.tile(diag_elem.T, (1, N))) / 2.
        S /= norm
    return S


def compute_K_map(train_ids, test_ids, docfeats, sim='linear', train_idx=[], normalize=False):
    """
    Imput:
        - train_ids, test_ids: list of document ids (keys of docfeats)
        - docfeats: dict with doc_id:feat
        - sim: type of similarity
        - train_idx: list of indexes for train_ids, e.g. for svms we don't need all training examples
    Returns:
        - kernel map of size len(test_ids)xlen(train_ids) with similarities of the test to the training docs
    """
    # if linear similarity or variant thereof, we're quicker with a dot product
    if sim in ('linear', 'cosine', 'angularsim', 'angulardist'):
        # transform features into matrix
        featmat_train, featurenames = features2mat(docfeats, train_ids)
        featmat_test, featurenames = features2mat(docfeats, test_ids, featurenames)
        # possibly normalize vectors
        if sim in ('cosine', 'angularsim', 'angulardist'):
            featmat_train /= np.linalg.norm(featmat_train, axis=0)
            featmat_test /= np.linalg.norm(featmat_test, axis=0)
        # compute kernel map
        K_map = featmat_test.dot(featmat_train.T).toarray()
        # transform further
        if sim in ('angularsim', 'angulardist'):
            # make sure values are in the correct range (roundoff errors)
            K_map = np.minimum(K_map, 1)
            K_map = np.maximum(K_map, 0)
            if sim == 'angularsim':
                K_map = 1. - 2. * np.arccos(K_map) / np.pi
            else:
                # distance
                K_map = 2. * np.arccos(K_map) / np.pi
    else:
        # compute similarity of all test examples to all training examples
        n_tr = len(train_ids)
        n_ts = len(test_ids)
        if not len(train_idx):
            train_idx = list(range(n_tr))
        K_map = np.array([[compute_sim(docfeats[did_ts], docfeats[train_ids[j]], sim)
                           for j in train_idx] for did_ts in test_ids])
    if normalize:
        train_sim = np.array([[compute_sim(docfeats[did_tr], docfeats[did_tr], sim) for did_tr in train_ids]])
        test_sim = np.array([[compute_sim(docfeats[did_ts], docfeats[did_ts], sim) for did_ts in test_ids]]).T
        norm = (np.tile(train_sim, (n_ts, 1)) + np.tile(test_sim, (1, n_tr))) / 2.
        K_map /= norm
    return K_map


def get_k_most_similar(K, row_ids, col_ids, did, k=10):
    """
    given the kernel matrix and corresponding docids, get the k most similar documents corresponding to the doc did

    Input:
        - K: kernel matrix or map with similarities (the higher the more similar)
        - row_ids: list of ids in the order of the rows in K
        - col_ids: list of ids in the order of the cols in K (same as row_ids in the case of the kernel matrix, not map)
        - did: a single docids corresponding to one row in K and to which similar documents should be found
        - k: how many similar documents should be returned
    Returns:
        - simdocids: [(docid, simscore)] a list of k docids and their respective similarity to the target did
    """
    try:
        dididx = row_ids.index(did)
    except ValueError:
        print("did must have a designated row in K")
        return []
    # get k indexes for K
    k_idx = np.flipud(np.argsort(K[dididx, :]))[:k + 1]
    return [(col_ids[i], K[dididx, i]) for i in k_idx if not col_ids[i] == did][:k]
