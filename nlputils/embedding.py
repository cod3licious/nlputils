from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform


def tsne_sim(S, no_dims=2, earlystop=True, init='random', verbose=True, max_iter=1500):
    """
    TSNE_Sim Performs symmetric t-SNE on similarity matrix S
       mappedX = tsne_sim(S, no_dims)

     The function performs symmetric t-SNE on pairwise similarity matrix S
     to create a low-dimensional map of no_dims dimensions (default = 2).
     The matrix S is assumed to be symmetric, sum up to 1, and have zeros
     on the diagonal.
     The low-dimensional data representation is returned in mappedX.
    """
    # Initialize some variables
    n = S.shape[0]                                     # number of instances
    momentum = 0.5                                     # initial momentum
    final_momentum = 0.8                               # value to which momentum is changed
    mom_switch_iter = 250                              # iteration at which momentum is changed
    exageration = 8.                                   # by how much we lie about the P-values
    stop_lying_iter = 100                              # iteration at which lying about P-values is stopped
    epsilon = 500.                                     # initial learning rate
    min_gain = .01                                     # minimum gain for delta-bar-delta
    # initialize the solution
    if init == 'kpca' and no_dims == 2:
        # kPCA initialization
        x, y = proj2d(S, use_tsne=False)
        mappedX = np.zeros((n, no_dims))
        mappedX[:, 0] = x
        mappedX[:, 1] = y
    else:
        mappedX = .0001 * np.random.randn(n, no_dims)
    y_incs = np.zeros(mappedX.shape)
    gains = np.ones(mappedX.shape)
    last_cost = np.inf
    # Make sure S-vals are set properly
    np.fill_diagonal(S, 0.)                            # set diagonal to zero
    S = 0.5 * (S + S.T)                                # symmetrize S-values
    S /= np.sum(S)                                     # make sure S-values sum to one
    S = np.maximum(S, 1e-12)
    const = np.sum(S * np.log(S))                      # constant in KL divergence
    S *= exageration                                   # lie about the S-vals to find better local minima
    # Run the iterations
    for itr in range(max_iter):
        # Compute joint probability that point i and j are neighbors
        # sum_mappedX = np.sum(np.square(mappedX), 1)
        # Student-t distribution
        # num = 1 / (1 + np.add(np.add(-2 * np.dot(mappedX, mappedX.T), sum_mappedX).T, sum_mappedX))
        num = 1 / (1 + squareform(pdist(mappedX, 'sqeuclidean')))
        np.fill_diagonal(num, 0.)                       # set diagonal to zero
        Q = num / np.sum(num)                             # normalize to get probabilities
        Q = np.maximum(Q, 1e-12)
        # Compute the gradients (faster implementation)
        L = (S - Q) * num
        # free some memory
        del num
        y_grads = 4. * np.dot((np.diag(np.sum(L, 0)) - L), mappedX)
        # Update the solution (note that the y_grads are actually -y_grads)
        gains = (gains + .2) * np.invert(np.sign(y_grads) == np.sign(y_incs)) + \
            (gains * .8) * (np.sign(y_grads) == np.sign(y_incs))
        gains[gains < min_gain] = min_gain
        y_incs = momentum * y_incs - epsilon * (gains * y_grads)
        mappedX += y_incs
        mappedX -= np.tile(np.mean(mappedX, 0), (n, 1))
        # Update the momentum if necessary
        if itr == mom_switch_iter:
            momentum = final_momentum
        if itr == stop_lying_iter:
            S /= exageration
        # Print out progress
        if not (itr + 1) % 25:
            if itr < stop_lying_iter:
                cost = const - np.sum(S / exageration * np.log(Q))
            else:
                cost = const - np.sum(S * np.log(Q))
                if earlystop and itr > mom_switch_iter and cost >= last_cost - 0.000001:
                    break
                else:
                    last_cost = cost
            if verbose:
                print('Iteration %i: error is %.5f' % (itr + 1, cost))
    return mappedX


def classical_scaling(K, nev=2, evcrit='LM'):
    """
    compute 2 eigenvalues and -vectors of a similarity matrix
    Input:
        - K: a symmetric nxn similarity matrix (for dissimilarity matrices, first multiply by -1/2)
        - nev: how many eigenvalues should be computed
        - evcrit: how eigenvalues should be selected ('LM' for largest real part, 'SM' for smallest real part)
    Return:
        - mappedX: n x 2 matrix of mapped data
    """
    n, m = K.shape
    H = np.eye(n) - np.tile(1. / n, (n, n))
    B = np.dot(np.dot(H, K), H)
    D, V = eigsh((B + B.T) / 2., k=nev, which=evcrit)  # guard against spurious complex e-vals from roundoff
    return np.dot(V.real, np.diag(np.sqrt(np.abs(D.real))))


def proj2d(K, use_tsne=True, evcrit='LM', max_iter=1500, verbose=True):
    """
    wrapper function to project data to 2D
    """
    if use_tsne:
        if verbose:
            print("performing tSNE: %i datapoints" % K.shape[0])
        X = tsne_sim(K, max_iter=max_iter, verbose=verbose)
        x = X[:, 0]
        y = X[:, 1]
    else:
        if verbose:
            print("performing classical scaling: %i datapoints" % K.shape[0])
        if evcrit == 'LM' or evcrit == 'SM':
            X = classical_scaling(K, evcrit=evcrit)
            x = X[:, 0]
            y = X[:, 1]
        elif evcrit == 'LS':
            x = classical_scaling(K, nev=1, evcrit='LM')
            y = classical_scaling(K, nev=1, evcrit='SM')
        else:
            print("ERROR: evcrit not known!")
            return None
    return x, y
