from __future__ import division
import numpy as np
import sys
from misc import invert_dict, norm_dict, select_copy

def knn(K_map, train_ids, test_ids, doccats, k=25, adapt=True, alpha=5, weight=True):
    """
    k nearest neighbors
    Input:
        K_map: matrix of size len(test_ids)xlen(train_ids) with similarities of the test to the training docs
        train_ids: list of document ids in the order used for K_map
        test_ids: list of document ids that need to be assigned a category
        doccats: true categories of training documents
        k (default=25): how many nearest neighbors of 1 category should be considered (at most)
        adapt (default=True): if the k value should be adapted (for skewed category distributions to avoid the bias of 
               categories with many samples)
        alpha (default=5): if adapt=True, how many samples should be considered at least
        weight (default=True): if the nearest neighbors should be weighted by their similarity
    Returns:
        for every test document a likeliness score for every category: dict[tid] = dict[cat]:score (between 0 and 1)
    """
    categories = sorted(invert_dict(doccats).keys())
    # select from doccats only training examples
    doccats = select_copy(doccats, train_ids)
    # dict with cat:[docs]
    cat_docs = invert_dict(doccats)
    for cat in categories:
        if not cat in cat_docs:
            cat_docs[cat] = []
    # k for every category
    if adapt:
        # max number of samples in category
        cat_max = max([len(cat_docs[cat]) for cat in categories])
        # adaptive k is at least alpha and at most k or number of samples in category
        k_cat = {cat: min(len(cat_docs[cat]),max(int(alpha),int(k*len(cat_docs[cat])/cat_max))) for cat in categories}
    else:
        k_cat = {cat:k for cat in categories}
    # get index for k nearest neighbors
    k_idx = np.fliplr(np.argsort(K_map))[:,:k]
    train_ids_dict = {doc:i for i, doc in enumerate(train_ids)}
    likely_cat = {did_ts:{} for did_ts in test_ids}
    for cat in categories:
        # get K_map index of all training documents that belong to the category
        cdoc_idx = set([train_ids_dict[doc] for doc in cat_docs[cat]])
        # compute the score for that category for every test example
        for i, did_ts in enumerate(test_ids):
            # get overlap between category specific training examples and the (adapted) k nearest neighbors of the test example
            tidx = sorted(set(cdoc_idx & set(k_idx[i,:k_cat[cat]])))
            if tidx and np.sum(K_map[i,tidx]):
                # get score
                if weight:
                    # sum of similarity of k nearest neighbors of category cat / sum of similarity of all knn
                    likely_cat[did_ts][cat] = np.sum(K_map[i,tidx])/np.sum(K_map[i,k_idx[i,:k_cat[cat]]])
                else:
                    # number of nearest neighbors of category cat
                    likely_cat[did_ts][cat] = float(len(tidx))/k_cat[cat]
            else:
                likely_cat[did_ts][cat] = 0.0
    # normalize score dicts - don't, they are already scaled between 0 and 1, w/o norm, thresholding works
    # for did_ts in test_ids:
    #     likely_cat[did_ts] = norm_dict(likely_cat[did_ts])
    return likely_cat

def get_labels(likely_cat, threshold='max'):
    """
    Input:
        likely_cat: the confidence scores for every category for every test doc as a dict (scores normalized to 1)
        threshold: threshold for the likeliness score: selects as many categories as have a score equal to or above the
                   threshold (between 0 and 1). If threshold is set to 'max', only the category with the highest score
                   is chosen (unless all categories have a score of 0)
    Returns:
        labels: dict with doc:[cats] for each test doc where the list contains as many categories as there are true cats
                --> the categories are chosen as the highest scoring categories in likely_cat
        confidence scores: dict with doc:[score] with confidence scores for every category in labels
    Note:
        if there is no likely category for a doc (i.e. the scores for all categories are 0.), the doc will get as a category
        '__UNKNOWN' with a confscores of 0.
    """
    categories = set(invert_dict(likely_cat).keys())
    labels = {}
    confscores = {}
    for tid in likely_cat:
        # sort categories by their likelihood
        catlist = sorted(categories,key=likely_cat[tid].get,reverse=True)
        scores = sorted(likely_cat[tid].values(), reverse=True)
        # scores for all categories are 0, so we don't have a clue
        if scores[0] == 0:
            ncat = 0
        # select categories above the threshold
        elif threshold == 'max':
            ncat = 1
        else:
            ncat = len([s for s in scores if s >= threshold])
        if ncat:
            labels[tid] = catlist[:ncat]
            # confidence score (same as likeliness score...)
            confscores[tid] = scores[:ncat]
        else:
            labels[tid] = ['__UNKNOWN'] # should decrease the # of false positives!
            confscores[tid] = [0.]
    return labels, confscores

def eval_classifier(test_ids, pred_cat, true_cat):
    """
    Input:
        test_ids: document ids of test documents, used as key for pred_cat and true_cat
        pred_cat: the predicted category(s) for every test doc as a dict {docid:[cats]}
        true_cat: dict with docid:[cats] - true categories for every test doc
    Returns:
        a dict with dicts with accuracy measures for the classifier for every category and a dict with total scores
        {cat:{"f1":x1, "pr":x2, "re":x3, "acc":x4}} - F1 score, precision, recall, accuracy
    Note:
        if a test sample can belong to multiple categories, each decision is considered on its own, i.e. if there are
        10 test samples, but each sample is associated with 3 categories, the _total accuracy will be the mean of 30 decisions
    """
    # total number of samples (= tp+fp+tn+fn)
    N = float(len(test_ids))
    # invert true cat dict to get categories and tids that should be selected for the cat
    categories = set(invert_dict(true_cat).keys())
    cat_true = invert_dict(select_copy(true_cat,test_ids))
    # the same for the predicted category labels
    cat_pred = invert_dict(pred_cat)
    for cat in categories:
        if not cat in cat_pred:
            cat_pred[cat] = []
        if not cat in cat_true:
            cat_true[cat] = []
    # look for overlap between cat_true and cat_pred
    cat_scores = {}
    tp_total = 0.
    tn_total = 0.
    fp_total = 0.
    fn_total = 0.
    for cat in categories:
        cat_scores[cat] = {}
        # true positive: cat true and guessed
        tp = float(len(set(cat_true[cat]) & set(cat_pred[cat])))
        tp_total += tp
        # false positive: cat guessed but not true
        fp = float(len(set(cat_pred[cat]).difference(set(cat_true[cat]))))
        fp_total += fp
        # false negative: cat true but not guessed
        fn = float(len(set(cat_true[cat]).difference(set(cat_pred[cat]))))
        fn_total += fn
        # true negatives: cat not true and not guessed
        tn = N - (tp+fp+fn)
        tn_total += tn
        if tp:
            # precision
            p = tp/(tp+fp)
            cat_scores[cat]["pr"] = p
            # recall (true positive rate)
            r = tp/(tp+fn)
            cat_scores[cat]["re"] = r
            # F1 score
            try:
                cat_scores[cat]["f1"] = (2.*p*r)/(p+r)
            except:
                cat_scores[cat]["f1"] = 0.
        else:
            cat_scores[cat]["pr"] = 0.
            cat_scores[cat]["re"] = 0.
            cat_scores[cat]["f1"] = 0.
        # false positive rate (for ROC)
        if fp:
            cat_scores[cat]["fpr"] = fp/(fp+tn)
        else:
            cat_scores[cat]["fpr"] = 0.
    # save total scores
    tot_scores = {}
    # precision
    p = tp_total/(tp_total+fp_total)
    tot_scores["pr"] = p
    # recall
    r = tp_total/(tp_total+fn_total)
    tot_scores["re"] = r
    # F1 score
    try:
        tot_scores["f1"] = (2.*p*r)/(p+r)
    except:
        tot_scores["f1"] = 0.
    # fpr
    tot_scores["fpr"] = fp_total/(fp_total+tn_total)
    return cat_scores, tot_scores
