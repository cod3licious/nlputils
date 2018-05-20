from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import object
import re
from math import log
from collections import Counter
from scipy.sparse import csr_matrix, dok_matrix
from unidecode import unidecode
from .dict_utils import norm_dict, invert_dict1, invert_dict2, select_copy


def norm_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_text(text, to_lower=True, norm_num=True):
    # clean the text: no fucked up characters, html, ...
    # if not isinstance(text, unicode):
    #     text = text.decode("utf-8")
    text = unidecode(text)
    text = re.sub(r"http(s)?://\S*", " ", text)  # remove links (other html crap is assumed to be removed by bs)
    if to_lower:
        text = text.lower()
    if norm_num:
        text = re.sub(r"[0-9]", "1", text)  # normalize numbers
    # clean out non-alphabet characters and normalize whitespace
    text = re.sub(r"[^A-Za-z0-9-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_bigram_scores(text_words, min_bgfreq=2.):
    """
    compute scores for identifying bigrams in a collection of texts
    compute unigram and bigram frequencies (of words) and give a score for every bigram.
    depending on this score (e.g. if higher than a threshold), bigram phrases can be identified
    in the raw texts before splitting them into words
    --> once the lists of words in the text_words are replaced by appropriate bigrams, the whole thing could be repeated
        to find trigrams, etc.
    Input:
        - text_words: a list (or generator) of lists of (preprocessed) words
        - min_bgfreq: how often a bigram has to have to occur in the corpus to be recognized (if both words occur just once
                      and that in combination, it still doesn't mean it's a true phrase because we have too little observations of the words
    Returns:
        - bigram_scores: a dict with bigrams ("word1 word2") and their score
    """
    unigram_freq = {}
    bigram_freq = {}
    for wordlist in text_words:
        for i, word in enumerate(wordlist):
            # count unigrams and bigrams
            try:
                unigram_freq[word] += 1.
            except:
                unigram_freq[word] = 1.
            if i:
                try:
                    bigram_freq["%s %s" % (wordlist[i - 1], word)] += 1.
                except:
                    bigram_freq["%s %s" % (wordlist[i - 1], word)] = 1.
    # compute bigram scores
    bigram_scores = {}
    for bigram in bigram_freq:
        # discount to ensure a word combination occurred a sufficient amount of times
        if max(0., bigram_freq[bigram] - min_bgfreq):
            bigram_scores[bigram] = bigram_freq[bigram] / \
                max(unigram_freq[bigram.split()[0]], unigram_freq[bigram.split()[1]])
    return bigram_scores


def find_bigrams(textdict, threshold=0.1):
    """
    find bigrams in the texts
    Input:
        - textdict: a dict with {docid: preprocessed_text}
        - threshold: for bigrams scores
    Returns:
        - bigrams: a list of "word1 word2" bigrams
    """
    docids = set(textdict.keys())
    # to identify bigrams, transform the texts into lists of words (assume texts are preprocessed)
    text_words = [textdict[did].split() for did in docids]
    bigram_scores = get_bigram_scores(text_words)
    return [bigram for bigram in bigram_scores if bigram_scores[bigram] > threshold]


def replace_bigrams(textdict, bigrams):
    """
    replace bigrams in the texts
    Input:
        - textdict: a dict with {docid: preprocessed_text}
        - bigrams: for bigrams scores
    Returns:
        - textdict: the same texts but preprocessed and with all bigrams joined as "word1_word2"
    """
    docids = set(textdict.keys())
    # substitute the bigrams in the texts - actually creates trigrams as well :)
    for did in docids:
        text = textdict[did]
        for bigram in bigrams:
            if bigram in text:
                text = text.replace(bigram, "%s_%s" % tuple(bigram.split()))
        textdict[did] = text
    return textdict


def compute_idf(docfeats):
    """
    Inputs:
        - docfeats: a dict with doc_id:{term:count}
    Returns:
        - Dw: a dict with {term: weight}
    """
    # total number of documents
    N = float(len(docfeats))
    # invert the dictionary to be term:{doc_id:count}
    termdocs = invert_dict2(docfeats)
    # compute idf for every term
    return norm_dict({term: log(N / len(termdocs[term])) for term in termdocs})


class FeatureTransform(object):
    """
    FeatureTransform

    a class to transform text into features (similar to sklearn classes)

    Usage:
        # initialize the FeatureTransformer
        ft = FeatureTransform(norm='max', weight=True, renorm='length', identify_bigrams=True)
        # use the training ids to compute the weights but transform all documents
        docfeats = ft.texts2features(textdict, trainids)
        # transform a set of new documents as well using the same weights and identified bigrams
        newdocfeats = ft.texts2features(newtextdict)

    Attributes:
        - to_lower: whether the text should be lower cased in preprocessing
        - norm_num: whether numbers should be normalized, e.g. both years '1989' and '2016' are transformed to '1111'
        - identify_bigrams: if bigrams should be found and replaced
        - norm (binary, max, length, sum, None): how the term counts for each doc should be normalized
        - weight: if idf term weights should be applied
        - renorm: how the features with applied weights should be renormalized
    """

    def __init__(self, norm='max', weight=True, renorm='length', identify_bigrams=True,
                 to_lower=True, norm_num=True, bg_threshold=0.1):
        self.norm = norm
        self.weight = weight
        self.renorm = renorm
        self.identify_bigrams = identify_bigrams
        self.to_lower = to_lower
        self.norm_num = norm_num
        self.bg_threshold = bg_threshold
        self.Dw = {}
        self.bigrams = []

    def texts2features(self, textdict, fit_ids=[]):
        """
        preprocess texts, count how often each word occurs, weight counts, normalize
        If this is called the first time, possibly the idf weights and bigrams are computed (using the documents
            specified in fit_ids), in future calls, the precomputed weights and bigrams are used, e.g. when applying
            the routine to new test documents.
        Input:
            - textdict: a dict with {docid: text}
            - fit_ids: if only a portion of all texts should be used to compute the weights and identify bigrams
                       (e.g. only training data - only used in the first initializing run)
        Returns:
            - docfeats: a dict with {docid: {term: (normalized/weighted) count}}
        """
        docids = set(textdict.keys())
        if not len(fit_ids):
            fit_ids = set(textdict.keys())
        # pre-process texts
        textdict_pp = {did: preprocess_text(textdict[did], self.to_lower, self.norm_num) for did in docids}
        # possibly find bigrams
        if self.identify_bigrams:
            if not self.bigrams:
                self.bigrams = find_bigrams(select_copy(textdict_pp, fit_ids), self.bg_threshold)
            textdict_pp = replace_bigrams(textdict_pp, self.bigrams)
        # split texts into tokens
        docfeats = {}
        for did in docids:
            featdict = dict(Counter(textdict_pp[did].split()))
            # normalize
            if self.norm:
                featdict = norm_dict(featdict, norm=self.norm)
            docfeats[did] = featdict
        # possibly compute idf weights and re-normalize
        if self.weight:
            if not self.Dw:
                self.Dw = compute_idf(select_copy(docfeats, fit_ids))
            for did in docids:
                # if the word was not in Dw (= not in the training set), delete it
                # (otherwise it can mess with renormalization)
                docfeats[did] = {term: docfeats[did][term] * self.Dw[term] for term in docfeats[did] if term in self.Dw}
        if self.renorm:
            for did in docids:
                docfeats[did] = norm_dict(docfeats[did], norm=self.renorm)
        return docfeats


def features2mat(docfeats, docids, featurenames=[]):
    """
    Transform a dictionary with features into a sparse matrix (e.g. for sklearn algorithms)
    Input:
        - docfeats: a dictionary with {docid:{word:count}}
        - docids: the subset of the docfeats (keys to the dict) that should be regarded,
                   defines rows of the feature matrix
        - featurenames: a list of words that define the columns of the feature matrix
                --> when the docids are the training samples, this can be left empty
                    and the words in the training document will be used; for testing give the returned names
    Returns:
        - featmat: a sparse matrix with docids x featurenames
        - featurenames: the list of words defining the columns of the featmat

    Example usage:
        features_train, featurenames = make_featmat(docfeats, trainids)
        features_test, featurenames = make_featmat(docfeats, testids, featurenames)
    """
    if not featurenames:
        featurenames = sorted(invert_dict1(select_copy(docfeats, docids)).keys())
    fnamedict = {feat: i for i, feat in enumerate(featurenames)}
    featmat = dok_matrix((len(docids), len(featurenames)), dtype=float)
    for i, did in enumerate(docids):
        for word in docfeats[did]:
            try:
                featmat[i, fnamedict[word]] = docfeats[did][word]
            except KeyError:
                pass
    featmat = csr_matrix(featmat)
    return featmat, featurenames
