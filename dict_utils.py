from __future__ import division
import numpy as np
from copy import deepcopy

def invert_dict0(adict):
    """
    for a key:value dict, return value:[key(s)],
    e.g. dict[doc] = label --> dict[label] = [docs]
    """
    inv_dict = {}
    [inv_dict.setdefault(v, []).append(k) for k, v in adict.iteritems()]
    return inv_dict

def invert_dict1(adict):
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

def select_copy(adict, key_ids):
    """
    make a copy of adict but including only the keys in key_ids
    """
    adict_copy = {}
    for key in key_ids:
        try:
            adict_copy[key] = deepcopy(adict[key])
        # ignore keys that are not in the original dict
        except KeyError:
            pass
    return adict_copy

def vec2dict(vector):
    """
    Input:
        vector: a list or (1d) numpy array
    Returns:
        a dict with {idx:value} for all values of the vector
    """
    return {i:val for i, val in enumerate(vector)}

def norm_dict(somedict, norm='max'):
    """
    Input:
        somedict: a dictionary with values of something
        norm (either 'max', 'sum', or 'length'): how to normalize the values
    Output:
        a dictionary with the normalized values
    """
    # can't normalize empty dicts
    if not somedict:
        return somedict
    if norm == 'binary':
        return {k:1. for k in somedict}
    elif norm == 'sum':
        N = float(sum(somedict.values()))
    elif norm == 'max':
        N = float(max(somedict.values()))
    elif norm == 'length':
        N = np.linalg.norm(somedict.values())
    elif norm == 'mean':
        N = np.mean(somedict.values())
    elif norm == 'std':
        N = np.std(somedict.values())
    else:
        print "ERROR: norm not known!!"
        return somedict
    if N:
        return {s:somedict[s]/N for s in somedict}
    else: 
        return somedict

