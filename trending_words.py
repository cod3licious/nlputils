from __future__ import division
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from nlputils.dict_utils import invert_dict0, invert_dict1


def trending_fun_tpr(tpr, fpr):
    # to get the development of word occurrences
    return tpr

def trending_fun_diff(tpr, fpr):
    # computes the trending score as the difference between tpr and fpr rate (not below 0 though)
    return np.maximum(tpr-fpr,0.)

def trending_fun_tprmean(tpr, fpr):
    # computes the trending score as the difference between tpr and fpr rate (not below 0 though)
    return 0.5*(tpr + np.maximum(tpr-fpr,0.))

def trending_fun_tprmult(tpr, fpr):
    # computes the trending score as the difference between tpr and fpr rate (not below 0 though)
    return tpr*np.maximum(tpr-fpr,0.)

def trending_fun_quot(tpr, fpr):
    # computes the trending score as the difference between tpr and fpr rate (not below 0 though)
    #return 1./(1.+np.exp(-tpr/np.maximum(fpr,sys.float_info.epsilon)))
    return (np.minimum(np.maximum(tpr/np.maximum(fpr,sys.float_info.epsilon),1.),4.)-1)/3.

def trending_fun_quotdiff(tpr, fpr):
    # computes the trending score as the difference between tpr and fpr rate (not below 0 though)
    #return 1./(1.+np.exp(-tpr/np.maximum(fpr,sys.float_info.epsilon)))
    return 0.5*(trending_fun_quot(tpr, fpr)+trending_fun_diff(tpr, fpr))


def get_k_trending_words(textdict, doccats, k=10, trending_fun=trending_fun_quotdiff, target_cats=[]):
    """
    For every category, using the texts belonging to this category vs. max of all other texts, 
    k 'distinguishing' words are found.
    Input:
        - textdict: a dict with {docid:text} (make a deepcopy if you like your original texts)
        - doccats: a dict with {docid:[cat(s)]} (to get trends in time, cat could also be a year)
        - k: how many words should be found for every category 
             --> if k is a list of words, only these words will be kept (if they occur)
        - trending_fun: which formula should be used when computing the score
        - target_cats: if not empty, only for some of the documents' categories trending words will be computed
    Returns:
        - trending_words: a dict with {cat:[(word1, score1), ..., (wordk,scorek)]}, 
          i.e. for every category k distinguishing words and a score indicating 
          how relevant the word is for this category (the higher the better)
    """
    # transform texts into sets of words
    text_words = {did: set(re.findall(r'[a-z_]+', textdict[did])) for did in textdict}
    # invert this dict to get for every word the documents it occurs in
    word_dids = {word:set(dids) for word, dids in invert_dict1(text_words).iteritems()}
    # invert the doccats dict to get for every category a list of documents belonging to it
    cats_dids = {cat:set(dids) for cat, dids in invert_dict1(doccats).iteritems()}
    # count the true positives for every word and category
    tpc_words = {}
    for word in word_dids:
        tpc_words[word] = {}
        for cat in cats_dids:
            tpc_words[word][cat] = len(cats_dids[cat].intersection(word_dids[word]))/len(cats_dids[cat])
    # possibly we are only interested in a subset of all categories
    if not target_cats:
        target_cats = cats_dids.keys()
    # for every category, compute a score for every word
    trending_words = {}
    for cat in target_cats:
        cat_wordscores = {}
        # compute a score for every word
        for word in word_dids:
            # in how many of the target category documents the word occurs
            tpr = tpc_words[word][cat]
            if tpr:
                # in how many of the non-target category documents the word occurs (mean+std)
                ## if the categories have drastically different sizes, maybe use
                # docids_cat = set(doccats.keys())
                # otherdocs = docids_cat.difference(cats_dids[cat])
                # fpr = len(otherdocs.intersection(word_dids[word]))/len(otherdocs)
                fprs = [tpc_words[word][c] for c in cats_dids if not c == cat]
                fpr = np.mean(fprs) + np.std(fprs)
                # compute score
                cat_wordscores[word] = trending_fun(tpr, fpr)
        # get the (at most) k relevant words for this category
        if type(k) == int:
            trending_words[cat] = [(word,cat_wordscores[word]) for word in sorted(cat_wordscores.keys(), key=cat_wordscores.get, reverse=True)[:min(k,len(cat_wordscores))]]
        else:
            trending_words[cat] = [(word,cat_wordscores[word]) if word in cat_wordscores else (word, 0.) for word in k]
    return trending_words

def get_trending_topics(textdict, trending_words_list, threshold=0.2):
    """
    Inputs:
        - textdict: a dict with {did: text} - most likely you want this to contain only the docs of the target cat
        - trending_words_list: a list with [(word: score)] like the trending words returned for a single category
    Returns:
        - trending_topics: a dict with {topic_id: [trending words]}, i.e. all trending words grouped to topics
        - topic_docs: a dict with {topic_id: did}, i.e. for every trending topic the best matching document
    """
    # get for every word the documents it occurs in
    text_words = {did: set(re.findall(r'[a-z_]+', textdict[did])) for did in textdict}
    word_dids = {word:set(dids) for word, dids in invert_dict1(text_words).iteritems()}
    # for simplicity
    words =[w[0] for w in trending_words_list]
    word_scores = {w[0]:w[1] for w in trending_words_list}
    # compute the similarity between all trending words
    similarities = {}
    for i, w1 in enumerate(words[:-1]):
        for w2 in words[i+1:]:
            # sim = # docs with w1 and w2 / min # docs with w1 or w2
            similarities[(w1,w2)] = len(word_dids[w1].intersection(word_dids[w2]))/min(len(word_dids[w1]),len(word_dids[w2]))
    # assign a topic to every word
    trending_topics = {i:[w] for i, w in enumerate(words)}
    # merge topics by median similarity
    while len(trending_topics) > 1:
        tpcs = sorted(trending_topics.keys())
        tpc_sims = {}
        for i, t1 in enumerate(tpcs[:-1]):
            for t2 in tpcs[i+1:]:
                # sim = median similarity of words in topics
                tpc_sims[(t1,t2)] = np.median([similarities[(w1,w2)] if (w1,w2) in similarities else similarities[(w2,w1)]
                                               for w1 in trending_topics[t1] for w2 in trending_topics[t2]])
        # merge the 2 most similar topics - if above threshold
        (t1,t2) = max(tpc_sims.keys(), key=tpc_sims.get)
        if tpc_sims[(t1,t2)] < threshold:
            break
        trending_topics[t1].extend(trending_topics[t2])
        del trending_topics[t2]
    # find the best doc for every topic
    topic_docs = {}
    for topic in trending_topics:
        doc_scores_temp = {}
        for w in trending_topics[topic]:
            for did in word_dids[w]:
                # weight by (normalized) # of occurrences of word?
                try:
                    doc_scores_temp[did] += word_scores[w]
                except:
                    doc_scores_temp[did] = word_scores[w]
        # take document with max score
        topic_docs[topic] = max(doc_scores_temp.keys(), key=doc_scores_temp.get)
    return trending_topics, topic_docs


def test_trending_computations(trending_fun=trending_fun_diff, fun_name='Rate difference'):
    """
    given a function to compute the "trending score" of a word given its true and false positive rate,
    plot the distribution of scores (2D) corresponding to the different tpr and fpr
    """
    # make a grid of possible tpr and fpr combinations
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
    test_trending_computations()
    test_trending_computations(trending_fun_tprmean, 'tpr mean')
    test_trending_computations(trending_fun_tprmult, 'tpr mult')
    test_trending_computations(trending_fun_quot, 'Rate quotient')
    test_trending_computations(trending_fun_quotdiff, 'Mean of Rate quotient and difference')
    plt.show()
