from __future__ import division
import re
import numpy as np
from nltk.stem import SnowballStemmer
from misc import clean_text, norm_dict

def contains_number(text):
    """
    checks if a text contains a number
    ...numbers can occur way more often than any other single word, so don't give them
    too much weight, otherwise their count will fuck with that of the normal words
    """
    #return len(re.findall(r"[0-9]+",text))
    if re.search(r"[0-9]+",text):
        return 1.
    else:
        return 0.

def _count_word(text,wordregex=r'[a-z]{2,}',stemming='',**kwargs):
    """
    Input:
        text: some text from which binary word features should be extracted
        wordregex (default=r'[a-z]{2,}'): what defines a word (e.g. only characters, min word length 2)
        stemming (default=None): string of language for which stemming should be performed on words (e.g. 'english','german')
    Returns:
        featdict: a dictionary including the words in the text as keys and their counts as values
    """
    wordlist = re.findall(wordregex,text)
    featdict = {}
    if stemming:
        stemmer = SnowballStemmer(stemming,ignore_stopwords=True)
    for word in wordlist:
        if stemming:
            word = stemmer.stem(word)
        try:
            featdict[word] += 1.
        except:
            featdict[word] = 1.
    return featdict

def _count_ngram(text, n_low=3, n_up=7, **kwargs):
    """
    Input:
        text: some (assumed to be clean) text from which binary n-gram features should be extracted
        n_low (default=3) - n_up (default=7): extracted n-grams will be of length n_low-n_up
    Returns:
        featdict: a dictionary including the n-grams in the text as keys and their counts as values
    """
    featdict = {}
    # count occurrences of all substrings
    for k in range(n_low, n_up+1):
        for i in range(len(text)-k+1):
            sub = text[i:i+k]
            try:
                featdict[sub] += 1.
            except:
                featdict[sub] = 1.
    # in case the text was shorter than n_low
    if not featdict:
        featdict[''] = 1.
    return featdict

def extract_features(text, feat='word', **kwargs):
    """
    Input:
        text: some text for which features should be computed
        feat (either 'word' or 'ngram'): whether to extract whole words or different length n-grams
        norm (either None, 'max', 'nwords', or 'length'): how to normalize the counts
        **kwargs: additional arguments for the underlying methods 
                  (wordregex and stemming for feat='word' or n_low and n_up for feat='ngram')
    Output:
        a dictionary with (possibly normalized) counts of the occurrences or words or n-grams in the text
    """
    # extract feature dict
    if feat == 'word':
        return _count_word(text,**kwargs)
    elif feat == 'ngram':
        return _count_ngram(text,**kwargs)
    else:
        print "ERROR: feat not known!!"
        return dict()

def getall_features(textdict, feat='word', norm='max', usenumber=True, **kwargs):
    """
    runs extract_features for a whole dict of docs
    Input:
        textdict: dict with doc_id:text
        feat (word or ngram): what kind of features should be computed
        norm (binary, max, length, nwords, None): how the term counts for each doc should be normalized
        usenumber (True or False): if numbers should be considered
    Returns:
        docfeats: dict with doc_id:{feature:count} (where the features depend on the parameters)
    """
    docfeats = {}
    for doc, text in textdict.iteritems():
        if usenumber:
            num = contains_number(text)
        # for ngrams, all the text has to be cleaned
        if feat == 'ngram':
            text = clean_text(text)
        # otherwise just replace apostrophes by empty strings to avoid splits there
        else:
            text = text.replace("'","")
            text = text.lower()
        docfeats[doc] = extract_features(text, feat, **kwargs)
        if usenumber and num:
                docfeats[doc]['__NUMBER'] = min(num,max(docfeats[doc].values()))
        if norm:
            if norm == 'binary':
                docfeats[doc] = {term:1. for term in docfeats[doc]}
            else:
                docfeats[doc] = norm_dict(docfeats[doc], norm=norm)
    return docfeats
