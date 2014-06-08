from __future__ import division
import sys
from copy import deepcopy
from math import log
import numpy as np
from misc import invert_dict, invert_dict2, norm_dict, select_copy

def _length(docfeats):
    """
    docfeats: a dict with doc_id:{term:count}
    """
    # invert the dictionary to be term:{doc_id:count}, all we need are the terms though
    termlist = set(invert_dict2(docfeats).keys())
    # compute number of characters for every term
    maxlen = float(max([len(term) for term in termlist]))
    Dw = {}
    for term in termlist:
        Dw[term] = len(term)/maxlen # already scaled
    return Dw

def _idf(docfeats):
    """
    docfeats: a dict with doc_id:{term:count}
    """
    # total number of documents
    N = float(len(docfeats))
    # invert the dictionary to be term:{doc_id:count}
    termdocs = invert_dict2(docfeats)
    # compute idf for every term
    Dw = {}
    for term in termdocs:
        Dw[term] = log(N/len(termdocs[term])) + 1.
    return norm_dict(Dw)

def _df(docfeats):
    """
    docfeats: a dict with doc_id:{term:count}
    """
    # total number of documents
    N = float(len(docfeats))
    # invert the dictionary to be term:{doc_id:count}
    termdocs = invert_dict2(docfeats)
    # compute df for every term
    Dw = {}
    for term in termdocs:
        Dw[term] = 1. - len(termdocs[term])/N
    return norm_dict(Dw)

def _saliency(docfeats):
    """
    docfeats: a dict with doc_id:{term:count}
    """
    # total number of documents
    N = float(len(docfeats))
    # invert the dictionary to be term:{doc_id:count}
    termdocs = invert_dict2(docfeats)
    # compute saliency for every term
    Dw = {}
    for term, countdict in termdocs.iteritems():
        # for all docs where the term occurs, just take variance,
        # for all others, the variance is mu^2 [since var = (0-mu)^2]
        Nt = len(termdocs[term])
        Dw[term] = np.var(countdict.values()) + (np.mean(countdict.values())**2)/(N-Nt)
    return norm_dict(Dw)

def _infogain(docfeats,doccats):
    """
    docfeats: a dict with doc_id:{term:count}
    doccats: a dict with doc_id:[categories]
    """
    # total number of documents
    N = float(len(docfeats))
    # invert document category dict to get dict[cat]:[docs]
    catdict = invert_dict(doccats)
    # do some precomputation for all categories
    catropy = -sum([(float(len(catdict[cat]))/N)*np.log2(len(catdict[cat])/N) for cat in catdict])
    # invert the docterm dictionary to be term:{doc_id:count}
    termdocs = invert_dict2(docfeats)
    Dw = {}
    for term in termdocs:
        # compute prob of term
        Nt = float(len(termdocs[term]))
        tprob = 2.*Nt/N
        # compute other stuff depending on category: Ntc/Nt * log(Ntc/Nt)
        tcprob = 0.
        for cat in catdict:
            Ntc = float(len(set(catdict[cat]) & set(termdocs[term].keys())))
            if Ntc:
                tcprob += (Ntc/Nt)*np.log2(Ntc/Nt)
        Dw[term] = catropy + tprob*tcprob
    return norm_dict(Dw)

def _odds(docfeats,doccats):
    """
    docfeats: a dict with doc_id:{term:count}
    doccats: a dict with doc_id:[categories]
    """
    # total number of documents
    N = float(len(docfeats))
    # set of all documents
    alldocs = set(docfeats.keys())
    # invert document category dict to get dict[cat]:[docs]
    catdict = invert_dict(doccats)
    # invert the docterm dictionary to be term:{doc_id:count}
    termdocs = invert_dict2(docfeats)
    Dw = {}
    for term in termdocs:
        # compute scores for every category
        scores = []
        for cat in catdict:
            # true positive: doc in cat and doc contains term
            tp = float(len(set(catdict[cat]) & set(termdocs[term].keys())))
            # true negative: doc not in cat and doc doesn't contain term
            tn = float(len(alldocs.difference(set(catdict[cat])) & alldocs.difference(set(termdocs[term].keys()))))
            if tp and tn:
                # false positive: doc in cat but doesn't contain term
                fp = max(1.,float(len(set(catdict[cat]))) - tp)
                # false negative: doc not in cat but contains term
                fn = max(1.,float(len(set(termdocs[term].keys()))) - tp)
                scores.append((tp*tn)/(fp*fn))
            else:
                scores.append(0.)
        # the weight for the term is the max of the scores
        Dw[term] = max(scores)
    return norm_dict(Dw)

def _modds(docfeats,doccats):
    """
    docfeats: a dict with doc_id:{term:count}
    doccats: a dict with doc_id:[categories]
    """
    # total number of documents
    N = float(len(docfeats))
    # set of all documents
    alldocs = set(docfeats.keys())
    # invert document category dict to get dict[cat]:[docs]
    catdict = invert_dict(doccats)
    # invert the docterm dictionary to be term:{doc_id:count}
    termdocs = invert_dict2(docfeats)
    Dw = {}
    for term in termdocs:
        # compute scores for every category
        tot_tp = 0.
        tot_tn = 0.
        tot_fp = 0.
        tot_fn = 0.
        for cat in catdict:
            Nc = len(set(catdict[cat]))
            # true positive: doc in cat and doc contains term
            tp = len(set(catdict[cat]) & set(termdocs[term].keys()))
            tot_tp += tp/Nc
            # true negative: doc not in cat and doc doesn't contain term
            tot_tn += float(len(alldocs.difference(set(catdict[cat])) & alldocs.difference(set(termdocs[term].keys()))))
            # false positive: doc in cat but doesn't contain term
            tot_fp += (float(len(set(catdict[cat]))) - tp)
            # false negative: doc not in cat but contains term
            tot_fn += (float(len(set(termdocs[term].keys()))) - tp)
        # the weight for the term is the max of the scores
        Dw[term] = (tot_tp*tot_tn)/(max(1.,tot_fp)*max(1.,tot_fn))
    return norm_dict(Dw)

def _cvarrange(docfeats,doccats,wtype='var'):
    """
    docfeats: a dict with doc_id:{term:count}
    doccats: a dict with doc_id:[categories]
    wtype: var, mdiff or range
    """
    # invert document category dict to get dict[cat]:[docs]
    catdict = invert_dict(doccats)
    # invert the docterm dictionary to be term:{doc_id:count}
    termdocs = invert_dict2(docfeats)
    Dw = {}
    for term in termdocs:
        # compute scores for every category
        scores = []
        for cat in catdict:
            # number of documents in the category
            Nc = float(len(catdict[cat]))
            # number of documents in the category that also contain term t
            Ntc = float(len(set(catdict[cat]) & set(termdocs[term].keys())))
            scores.append(Ntc/Nc)
        # the weight for the term is either the variance, range, or max diff from mean
        if wtype == 'var':
            Dw[term] = np.var(scores)
        elif wtype == 'range':
            Dw[term] = np.max(scores) - np.min(scores)
        else:
            mscore = np.mean(scores)
            Dw[term] = max([np.abs(mscore-s) for s in scores])
    return norm_dict(Dw)

def apply_weights(docfeats,Dw,renorm=None):
    """
    Input:
        docfeats: a dict with doc_id:features where features is a dict of term:counts
        Dw: a dictionary with term:weight
        renorm: if the features should be renormalized after applying the weights ('max' or 'length')
    Returns:
        docfeats with doc_id:{term:(count*weight)}
    """
    if Dw:
        docfeats_temp = deepcopy(docfeats)
        for doc in docfeats_temp:
            for term in docfeats_temp[doc]:
                try:
                    docfeats_temp[doc][term] *= Dw[term]
                except:
                    docfeats_temp[doc][term] = sys.float_info.epsilon
            if renorm:
                docfeats_temp[doc] = norm_dict(docfeats_temp[doc],norm=renorm)
        return docfeats_temp
    else:
        return docfeats

def _weights_wrapper(docfeats, doccats={}, weight='idf', doc_ids=[]):
    """
    Input:
        docfeats: a dict with doc_id:features where features is a dict of term:counts
        doccats: a dict with doc_id:[categories]
        weight: the kind of weights that should be computed:
            without doccats: 'length', 'idf', 'df', or 'saliency'
            with doccats: 'infogain', 'odds', 'cvar', 'crange'
        doc_ids: which docs in docfeats (and doccats) should be used to compute the weights
    Returns:
        Dw: a dictionary with term:weight
    """
    # possibly term weights are only supposed to be computed using a subset of the given docs (e.g. xval)
    if doc_ids:
        docfeats = select_copy(docfeats, doc_ids)
        if doccats:
            doccats = select_copy(doccats, doc_ids)
    # compute weight
    if weight == 'length':
        return _length(docfeats)
    elif weight == 'idf':
        return _idf(docfeats)
    elif weight == 'df':
        return _df(docfeats)
    elif weight == 'saliency':
        return _saliency(docfeats)
    # class dependent weights
    elif weight in ('infogain','odds','modds','cvar','crange','cmdiff'):
        if not doccats:
            print "ERROR: need doccats for weight depending on categories"
            return {}
        if weight == 'infogain':
            return _infogain(docfeats,doccats)
        elif weight == 'odds':
            return _odds(docfeats,doccats)
        elif weight == 'modds':
            return _modds(docfeats,doccats)
        elif weight == 'cvar':
            return _cvarrange(docfeats,doccats,wtype='var')
        elif weight == 'crange':
            return _cvarrange(docfeats,doccats,wtype='range')
        elif weight == 'cmdiff':
            return _cvarrange(docfeats,doccats,wtype='mdiff')
    else:
        print "ERROR: weight not known!!"
        return {}

def get_weights(docfeats, doccats={}, weights=['idf'], doc_ids=[], combine='mean'):
    """
    Input:
        docfeats: a dict with doc_id:features where features is a dict of term:counts
        doccats: a dict with doc_id:[categories]
        weights: the kinds of weights that should be computed:
            without doccats: 'length', 'idf', 'df', or 'saliency'
            with doccats: 'infogain', 'odds', 'cvar', 'crange'
        doc_ids: which docs in docfeats (and doccats) should be used to compute the weights
        combine: how the different weights should be combined (product or mean)
    Returns:
        Dw: a dictionary with term:weight where weight is a product/mean of all the chosen weights
    """
    if not len(weights):
        return {}
    # possibly term weights are only supposed to be computed using a subset of the given docs (e.g. xval)
    if doc_ids:
        docfeats = select_copy(docfeats, doc_ids)
        if doccats:
            doccats = select_copy(doccats, doc_ids)
    # compute all the weights
    if len(weights) == 1:
        return _weights_wrapper(docfeats, doccats, weights[0])
    else:
        Dw_list = []
        for w in weights:
            Dw_list.append(_weights_wrapper(docfeats, doccats, w))
        Dw = {}
        if combine == 'mean':
            for term in Dw_list[0]:
                Dw[term] = np.mean([Dw_list[i][term] for i in range(len(Dw_list))])
        elif combine == 'sum':
            for term in Dw_list[0]:
                Dw[term] = sum([Dw_list[i][term] for i in range(len(Dw_list))])
        elif combine == 'max':
            for term in Dw_list[0]:
                Dw[term] = max([Dw_list[i][term] for i in range(len(Dw_list))])
        elif combine == 'product':
            for term in Dw_list[0]:
                Dw[term] = Dw_list[0][term]
                for i in range(1,len(Dw_list)):
                    Dw[term] *= Dw_list[i][term]
        elif type(combine) == list and len(combine) == len(Dw_list):
            for term in Dw_list[0]:
                Dw[term] = sum([Dw_list[i][term]*combine[i] for i in range(len(Dw_list))])
        else:
            print "ERROR: combine not known!!"
            return Dw_list[0]
        return Dw
