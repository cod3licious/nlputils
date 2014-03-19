import colorsys
import json
from scipy.sparse.linalg import eigsh
import numpy as np
from misc import invert_dict

def tsne_sim(S, no_dims=2):
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
    stop_lying_iter = 100                              # iteration at which lying about P-values is stopped
    max_iter = 1000                                    # maximum number of iterations
    epsilon = 500.                                     # initial learning rate
    min_gain = .01                                     # minimum gain for delta-bar-delta   
    # Make sure S-vals are set properly
    np.fill_diagonal(S,0.)                             # set diagonal to zero
    S = 0.5 * (S + S.T)                                # symmetrize S-values
    S /= np.sum(S)                                     # make sure S-values sum to one
    S = np.maximum(S, 1e-12)
    const = np.sum(S * np.log(S))                      # constant in KL divergence
    S *= 4.                                            # lie about the S-vals to find better local minima    
    # Initialize the solution
    mappedX = .0001 * np.random.randn(n, no_dims)
    y_incs  = np.zeros(mappedX.shape)
    gains = np.ones(mappedX.shape)
    # Run the iterations
    for itr in range(max_iter):       
        # Compute joint probability that point i and j are neighbors
        sum_mappedX = np.sum(np.square(mappedX), 1)
        # Student-t distribution
        num = 1 / (1 + np.add(np.add(-2 * np.dot(mappedX, mappedX.T), sum_mappedX).T, sum_mappedX))
        np.fill_diagonal(num,0.)                        # set diagonal to zero
        Q = num/np.sum(num)                             # normalize to get probabilities
        Q = np.maximum(Q, 1e-12)        
        # Compute the gradients (faster implementation)
        L = (S - Q) * num;
        y_grads = 4. * np.dot((np.diag(np.sum(L, 0)) - L), mappedX)            
        # Update the solution (note that the y_grads are actually -y_grads)
        gains = (gains + .2) * np.invert(np.sign(y_grads) == np.sign(y_incs)) + (gains * .8) * (np.sign(y_grads) == np.sign(y_incs))
        gains[gains < min_gain] = min_gain
        y_incs = momentum * y_incs - epsilon * (gains * y_grads)
        mappedX = mappedX + y_incs
        mappedX -= np.tile(np.mean(mappedX, 0),(n,1))        
        # Update the momentum if necessary
        if itr == mom_switch_iter:
            momentum = final_momentum
        if itr == stop_lying_iter:
            S /= 4.        
        # Print out progress
        if not (itr+1)%25:
            if itr < stop_lying_iter:
                cost = const - np.sum(S/4. * np.log(Q))
            else:
                cost = const - np.sum(S * np.log(Q))
            print 'Iteration %i: error is %.5f'%(itr+1, cost)
    return mappedX

def classical_scaling(K, nev=2, evcrit='LM'):
    """
    compute 2 eigenvalues and -vectors of a similarity matrix
    Input:
        K: a symmetric nxn similarity matrix (for dissimilarity matrices, first multiply by -1/2)
        nev: how many eigenvalues should be computed
        evcrit: how eigenvalues should be selected ('LM' for largest real part, 'SM' for smallest real part)
    Return:
        D: a vector containing the 2 eigenvalues of K
        V: the eigenvectors corresponding to D 
    """
    n, m = K.shape
    H = np.eye(n) - np.tile(1./n,(n,n))
    B = np.dot(np.dot(H,K),H)
    D, V = eigsh((B+B.T)/2.,k=nev,which=evcrit) # guard against spurious complex e-vals from roundoff
    return D.real, V.real

def get_colors(N=100):
    HSV_tuples = [(x*1.0/N, 1., 0.8) for x in range(N)]
    return map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

def colorindex(X, N=100):
    """
    X: list/vector for which different colors should be assigned
    N: number of colors that can be used
    """
    # get min & max values
    minv, maxv = min(X), max(X)
    # get a linear scale of the values
    lscale = np.linspace(minv, maxv, N)
    # for each value in X, find the index in the linspace
    return [np.nonzero(lscale>=val)[0][0] for val in X]

def pretty_coloring(X, varcol=0, N=100):
    """
    X: rows = observations, cols = variables
    varcol: variable for which different colors should be assigned
    N: number of colors that can be used
    """
    coloridx = colorindex(X[:,varcol], N)
    colors = np.array(get_colors(N))
    return colors[coloridx,:]

def prepare_viz(doc_ids, docdict, doccats, K, catdesc={}, use_tsne=True, filepath='docs.json', evcrit='LM'):
    """
    function to prepare text data for 2 dim visualization by saving a json file, that is a list of dicts,
    where each dict decodes 1 doc with "id" (doc_id), "x" and "y" (2dim coordinates derived from the kernel matrix
    using classical scaling), "title" (category/ies), "description" (whatever is in docdict at doc_id), "color" (for cat)
    Input:
        doc_ids: list with keys for docdict and doccats
        docdict: dict with docid:'description'
        doccats: dict with docid:cat
        K: kernel/similarity matrix in the order of doc_ids
        catdesc: category descriptions
        use_tsne: if tsne instead of classical_scaling should be used to get 2d representations (default)
        filepath: where the json file will be saved
        evcrit: how eigenvalues should be selected ('LM' for largest real part, 'SM' for smallest real part, 
                'LS' for 1 large and 1 small one)
    """
    # pretty preprocessing
    categories = set(invert_dict(doccats).keys())
    if not catdesc:
        catdesc = {cat:cat for cat in categories}    
    colorlist = get_colors(len(categories))
    colordict = {cat:(255*colorlist[i][0],255*colorlist[i][1],255*colorlist[i][2]) for i, cat in enumerate(sorted(categories))}
    # get x and y for visualization
    if use_tsne:
        print "performing tSNE"
        X = tsne_sim(K)
        x = X[:,0]
        y = X[:,1]
    else:
        print "performing classical scaling"
        if evcrit=='LM' or evcrit=='SM':
            D, X = classical_scaling(K, evcrit=evcrit)
            x = X[:,0]
            y = X[:,1]
        elif evcrit=='LS':
            D, x = classical_scaling(K, nev=1, evcrit='LM')
            D, y = classical_scaling(K, nev=1, evcrit='SM')
        else:
            print "ERROR: evcrit not known!"
            return None
    # save as json
    print "saving json"
    data_json = []
    for i, key in enumerate(doc_ids):
        data_json.append({"id":key,"x":x[i],"y":y[i],"title":str(key)+" (%s)"%catdesc[doccats[key][0]],"description":docdict[key],"color":"rgb(%i,%i,%i)"%colordict[doccats[key][0]]})
    with open(filepath,"w") as f:
        f.write(json.dumps(data_json,indent=2))
