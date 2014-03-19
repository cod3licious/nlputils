from __future__ import division
import numpy as np
import re
from copy import deepcopy

def invert_dict(adict):
    """
    for a key:[value(s)] dict, return value:[key(s)],
    e.g. dict[doc] = [terms] --> dict[term] = [docs]
    """
    inv_dict = {}
    [inv_dict.setdefault(v, []).append(k) for k, vlist in adict.iteritems() for v in vlist]
    return inv_dict

def invert_dict2(adict):
    """
    for a dict(key:dict(key2:value)), return dict(key2:dict(key:value)),
    e.g. dict[doc] = dict[term]:count --> dict[term] = dict[doc]:count
    """
    inv_dict = {}
    [inv_dict.setdefault(key2, {}).update({key:v}) for key, dict2 in adict.iteritems() for key2, v in dict2.iteritems()]
    return inv_dict

def vec2dict(vector):
    """
    Input:
        vector: a numpy array
    Returns:
        a dict with {idx:value} for all values of the vector
    """
    return {i:val for i, val in enumerate(vector)}

def norm_dict(somedict, norm='max'):
    """
    Input:
        somedict: a dictionary with values of something
        norm (either 'max', 'nwords', or 'length'): how to normalize the values
    Output:
        a dictionary with the normalized values
    """
    if norm == 'binary':
        return {k:1. for k in somedict}
    elif norm == 'nwords':
        N = sum(somedict.values())
    elif norm == 'max':
        N = max(somedict.values())
    elif norm == 'length':
        N = np.linalg.norm(somedict.values())
    else:
        print "ERROR: norm not known!!"
        return somedict
    if N:
        somedict_temp = deepcopy(somedict)
        for t in somedict_temp:
            somedict_temp[t] /= float(N)
        return somedict_temp
    else: 
        return somedict

def select_copy(adict, key_ids):
    """
    make a copy of adict but including only the keys in key_ids
    """
    adict_copy = {}
    for key in key_ids:
        adict_copy[key] = deepcopy(adict[key])
    return adict_copy

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
    

def clean_text(text):
    """
    very basic method to clean some text by removing everything that is not a character
    and converting the text to lower case --> for specific texts, you should probably write your own
    """
    web = re.compile("http(s)?://\S*")
    text = web.sub(" ", text) # remove links
    text = text.lower()   # set to lowercase
    text = text.replace("'","") # replace apostrophes by empty strings (don't want blanks there)
    text = re.sub(r"[^a-z]"," ", text) # remove all invalid characters
    text = re.sub(r'\s+', " ", text) # normalize whitespace
    return text.strip()

