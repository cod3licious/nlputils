from __future__ import unicode_literals, division, print_function, absolute_import
import numpy as np
from .dict_utils import invert_dict1, select_copy


def knn(K_map, train_ids, test_ids, doccats, k=25, adapt=True, alpha=5, weight=True):
    """
    k nearest neighbors
    Input:
        - K_map: matrix of size len(test_ids)xlen(train_ids) with similarities of the test to the training docs
        - train_ids: list of document ids in the order used for K_map
        - test_ids: list of document ids that need to be assigned a category
        - doccats: true categories of training documents (as a list, since multiple categories per document are allowed)
        - k (default=25): how many nearest neighbors of 1 category should be considered (at most)
        - adapt (default=True): if the k value should be adapted (for skewed category distributions to avoid the bias of
                categories with many samples)
        - alpha (default=5): if adapt=True, how many samples should be considered at least
        - weight (default=True): if the nearest neighbors should be weighted by their similarity
    Returns:
        - for every test document a likeliness score for every category: dict[tid] = dict[cat]:score (between 0 and 1)
    """
    categories = sorted(invert_dict1(doccats).keys())
    # select from doccats only training examples
    doccats = select_copy(doccats, train_ids)
    # dict with cat:[docs]
    cat_docs = invert_dict1(doccats)
    for cat in categories:
        if not cat in cat_docs:
            cat_docs[cat] = []
    # k for every category
    if adapt:
        # max number of samples in category
        cat_max = max([len(cat_docs[cat]) for cat in categories])
        # adaptive k is at least alpha and at most k or number of samples in category
        k_cat = {cat: min(len(cat_docs[cat]), max(int(alpha), int(k * len(cat_docs[cat]) / cat_max))) for cat in categories}
    else:
        k_cat = {cat: k for cat in categories}
    # get index for k nearest neighbors
    k_idx = np.fliplr(np.argsort(K_map))[:, :k]
    train_ids_dict = {doc: i for i, doc in enumerate(train_ids)}
    likely_cat = {did_ts: {} for did_ts in test_ids}
    for cat in categories:
        # get K_map index of all training documents that belong to the category
        cdoc_idx = set([train_ids_dict[doc] for doc in cat_docs[cat]])
        # compute the score for that category for every test example
        for i, did_ts in enumerate(test_ids):
            # get overlap between category specific training examples and the
            # (adapted) k nearest neighbors of the test example
            tidx = sorted(set(cdoc_idx & set(k_idx[i, :k_cat[cat]])))
            if tidx and np.sum(K_map[i, tidx]):
                # get score
                if weight:
                    # sum of similarity of k nearest neighbors of category cat / sum of similarity of all knn
                    likely_cat[did_ts][cat] = np.sum(K_map[i, tidx]) / np.sum(K_map[i, k_idx[i, :k_cat[cat]]])
                else:
                    # number of nearest neighbors of category cat
                    likely_cat[did_ts][cat] = float(len(tidx)) / k_cat[cat]
            else:
                likely_cat[did_ts][cat] = 0.
    return likely_cat


def get_labels(likely_cat, threshold='max'):
    """
    Input:
        - likely_cat: the confidence scores for every category for every test doc as a dict (scores normalized to 1)
        - threshold: threshold for the likeliness score: selects as many categories as have a score equal to or above the
                   threshold (between 0 and 1). If threshold is set to 'max', only the category with the highest score
                   is chosen (unless all categories have a score of 0)
    Returns:
        - labels: dict with doc:[cats] for each test doc where the list contains as many categories as have scores above threshold
                 --> the categories are chosen as the highest scoring categories in likely_cat
    Note:
        if threshold='max' and all categories have score of 0, a random category is chosen,
        otherwise if threshold is a float and no category has a score above threshold, the list will be empty
    """
    labels = {}
    for tid in likely_cat:
        # either take the most likely category
        if threshold == 'max':
            labels[tid] = [max(likely_cat[tid].keys(), key=likely_cat[tid].get)]
        # or all categories with a score above threshold
        else:
            labels[tid] = [cat for cat in likely_cat[tid] if likely_cat[tid][cat] >= threshold]
    return labels
