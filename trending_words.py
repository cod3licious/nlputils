from __future__ import division
import sys
import re
import numpy as np
from nlputils.dict_utils import invert_dict0, invert_dict1
from nlputils.preprocessing import preprocess_text, find_bigrams, replace_bigrams


def trending_fun_tpr(tpr, fpr):
    # to get the development of word occurrences
    return tpr

def trending_fun_diff(tpr, fpr):
    # computes the trending score as the difference between tpr and fpr rate (not below 0 though)
    return np.maximum(tpr-fpr, 0.)

def trending_fun_tprmean(tpr, fpr):
    # computes the trending score as the mean between the tpr and the difference between tpr and fpr rate
    return 0.5*(tpr + np.maximum(tpr-fpr, 0.))

def trending_fun_tprmult(tpr, fpr):
    return tpr*np.maximum(tpr-fpr, 0.)

def trending_fun_quot(tpr, fpr):
    #return 1./(1.+np.exp(-tpr/np.maximum(fpr,sys.float_info.epsilon)))
    return (np.minimum(np.maximum(tpr/np.maximum(fpr, sys.float_info.epsilon), 1.), 4.)-1)/3.

def trending_fun_quotdiff(tpr, fpr):
    #return 1./(1.+np.exp(-tpr/np.maximum(fpr,sys.float_info.epsilon)))
    return 0.5*(trending_fun_quot(tpr, fpr) + trending_fun_diff(tpr, fpr))


def get_trending_words(textdict, doccats, trending_fun=trending_fun_quotdiff, target_cats=[]):
    """
    For every category, using the texts belonging to this category vs. max of all other texts, 
    k 'distinguishing' words are found.
    Input:
        - textdict: a dict with {docid: text}
        - doccats: a dict with {docid: cat} (to get trends in time, cat could also be a year/day/week)
        - trending_fun: which formula should be used when computing the score
        - target_cats: if not empty, only for some of the documents' categories trending words will be computed
    Returns:
        - trending_words: a dict with {cat: {word: score}}, 
          i.e. for every category the words and a score indicating 
          how relevant the word is for this category (the higher the better)
          you could then do sorted(trending_words[cat], key=trending_words[cat].get, reverse=True)[:10]
          to get the 10 most distinguishing words for that category
    """
    # preprocess the texts incl. identifying bigrams
    textdict_pp = {did: preprocess_text(textdict[did], to_lower=True, norm_num=False) for did in textdict}
    textdict_pp = replace_bigrams(textdict_pp, find_bigrams(textdict_pp))
    # transform texts into sets of words
    text_words = {did: set(re.findall(r'[a-z0-9_-]+', textdict_pp[did])) for did in textdict_pp}
    # invert this dict to get for every word the documents it occurs in
    word_dids = {word: set(dids) for word, dids in invert_dict1(text_words).iteritems()}
    # invert the doccats dict to get for every category a list of documents belonging to it
    cats_dids = {cat: set(dids) for cat, dids in invert_dict0(doccats).iteritems()}
    # count the true positives for every word and category
    tpc_words = {}
    for word in word_dids:
        tpc_words[word] = {}
        for cat in cats_dids:
            # out of all docs in this category, in how many did the word occur?
            tpc_words[word][cat] = len(cats_dids[cat].intersection(word_dids[word]))/len(cats_dids[cat])
    # possibly we are only interested in a subset of all categories
    if not target_cats:
        target_cats = cats_dids.keys()
    # for every category, compute a score for every word
    trending_words = {}
    for cat in target_cats:
        trending_words[cat] = {}
        # compute a score for every word
        for word in word_dids:
            # in how many of the target category documents the word occurs
            tpr = tpc_words[word][cat]
            if tpr:
                # in how many of the non-target category documents the word occurs (mean+std)
                fprs = [tpc_words[word][c] for c in cats_dids if not c == cat]
                fpr = np.mean(fprs) + np.std(fprs)
                # compute score
                trending_words[cat][word] = trending_fun(tpr, fpr)
    return trending_words

def test_trending_computations(trending_fun=trending_fun_diff, fun_name='Rate difference'):
    """
    given a function to compute the "trending score" of a word given its true and false positive rate,
    plot the distribution of scores (2D) corresponding to the different tpr and fpr
    """
    # make a grid of possible tpr and fpr combinations
    import matplotlib.pyplot as plt
    x, y = np.linspace(0,1,101), np.linspace(1,0,101)
    fpr, tpr = np.meshgrid(x, y)
    score = trending_fun(tpr,fpr)
    plt.figure()
    plt.imshow(score)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xticks(np.linspace(0,101,11),np.linspace(0,1,11))
    plt.yticks(np.linspace(0,101,11),np.linspace(1,0,11))
    plt.title('Trending Score using %s'%fun_name)
    plt.colorbar()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_trending_computations()
    test_trending_computations(trending_fun_tprmean, 'tpr mean')
    test_trending_computations(trending_fun_tprmult, 'tpr mult')
    test_trending_computations(trending_fun_quot, 'Rate quotient')
    test_trending_computations(trending_fun_quotdiff, 'Mean of Rate quotient and difference')
    plt.show()
