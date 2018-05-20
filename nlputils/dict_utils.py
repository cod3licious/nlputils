from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np
from copy import deepcopy


def invert_dict0(adict):
    """
    for a key:value dict, return value:[key(s)],
    e.g. dict[doc] = label --> dict[label] = [docs]
    """
    inv_dict = {}
    [inv_dict.setdefault(v, []).append(k) for k, v in adict.items()]
    return inv_dict


def invert_dict1(adict):
    """
    for a key:[value(s)] dict, return value:[key(s)],
    e.g. dict[doc] = [terms] --> dict[term] = [docs]
    """
    inv_dict = {}
    [inv_dict.setdefault(v, []).append(k) for k, vlist in adict.items() for v in vlist]
    return inv_dict


def invert_dict2(adict):
    """
    for a dict(key:dict(key2:value)), return dict(key2:dict(key:value)),
    e.g. dict[doc] = dict[term]:count --> dict[term] = dict[doc]:count
    """
    inv_dict = {}
    [inv_dict.setdefault(key2, {}).update({key: v}) for key, dict2 in adict.items()
     for key2, v in dict2.items()]
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
    return {i: val for i, val in enumerate(vector)}


def norm_dict(somedict, norm='max', n_all=0):
    """
    Input:
        somedict: a dictionary with values of something
        norm (either 'max', 'sum', or 'length'): how to normalize the values
        n_all: for 'mean' and 'std' we need to know the total number of values
               (if the dict is sparse, zeros are not included)
    Output:
        a dictionary with the normalized values
    """
    # can't normalize empty dicts
    if not somedict:
        return somedict
    if norm == 'binary':
        return {k: 1. for k in somedict}
    elif norm == 'sum':
        N = float(sum(somedict.values()))
    elif norm == 'max':
        N = float(max(np.abs(list(somedict.values()))))
    elif norm == 'length':
        N = np.linalg.norm(list(somedict.values()))
    elif norm == 'mean':
        N = np.mean(list(somedict.values()) + [0.] * (n_all - len(somedict)))
    elif norm == 'std':
        N = np.std(list(somedict.values()) + [0.] * (n_all - len(somedict)))
    else:
        print("ERROR: norm not known!!")
        return somedict
    if N:
        return {s: somedict[s] / N for s in somedict}
    else:
        return somedict


def combine_dicts(a, b, op=max):
    """
    Input:
        a, b: dictionaries with values of something
        op: how the values should be combined (max or sum (pass the function, not a string!))
    Output:
        a dictionary with all values from both dictionary. if a value occurred in only one of the
            dicts it is returned as it was, otherwise the values corresponding to a key from both
            dicts are combined according to op (e.g. add values together or get max of all values)
    """
    return dict(list(a.items()) + list(b.items()) + [(k, op([a[k], b[k]])) for k in set(b) & set(a)])
