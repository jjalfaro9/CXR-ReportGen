#!/usr/bin/env python
# From: https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/cider/cider_scorer.py
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

import copy
import math
from collections import defaultdict
from typing import List, AnyStr, DefaultDict
import pickle

import numpy as np

def precook(s: AnyStr, n=4, out=False) -> DefaultDict:
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs: List[AnyStr], n=4) -> List[DefaultDict]: ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test: AnyStr, n=4) -> DefaultDict:
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)

class CiderScorer:

    def __init__(self, docTf, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        with open(docTf, 'rb') as f:
            data = pickle.load(f)
            self.document_frequency = data['document_frequency']
            self.ref_len = np.log(float(data['ref_len']))

    def counts2vec(self, cnts):
        """
        Function maps counts of ngram to vector of tfidf weights.
        The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
        The n-th entry of array denotes length of n-grams.
        :param cnts:
        :return: vec (array of dict), norm (array of float), length (int)
        """
        vec = [defaultdict(float) for _ in range(self.n)]
        length = 0
        norm = [0.0 for _ in range(self.n)]
        for (ngram,term_freq) in cnts.items():
            # give word count 1 if it doesn't appear in reference corpus
            df = np.log(max(1.0, self.document_frequency[ngram]))
            # ngram index
            n = len(ngram)-1
            # tf (term_freq) * idf (precomputed idf) for n-grams
            vec[n][ngram] = float(term_freq)*(self.ref_len - df)
            # compute norm for the vector.  the norm will be used for computing similarity
            norm[n] += pow(vec[n][ngram], 2)

            if n == 1:
                length += term_freq
        norm = [np.sqrt(n) for n in norm]
        return vec, norm, length

    def sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
        '''
        Compute the cosine similarity of two vectors.
        :param vec_hyp: array of dictionary for vector corresponding to hypothesis
        :param vec_ref: array of dictionary for vector corresponding to reference
        :param norm_hyp: array of float for vector corresponding to hypothesis
        :param norm_ref: array of float for vector corresponding to reference
        :param length_hyp: int containing length of hypothesis
        :param length_ref: int containing length of reference
        :return: array of score for each n-grams cosine similarity
        '''
        delta = float(length_hyp - length_ref)
        # measure consine similarity
        val = np.array([0.0 for _ in range(self.n)])
        for n in range(self.n):
            # ngram
            for (ngram,count) in vec_hyp[n].items():
                # vrama91 : added clipping
                val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

            if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                val[n] /= (norm_hyp[n]*norm_ref[n])

            assert(not math.isnan(val[n]))
            # vrama91: added a length based gaussian penalty
            val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
        return val

    def compute_cider(self, ctest, crefs):
        scores = []
        for test, refs in zip(ctest, crefs):
            # compute vector for test captions
            vec, norm, length = self.counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = self.counts2vec(ref)
                score += self.sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, test, refs):
        crefs = cook_refs(refs)
        ctest = cook_test(test)
        # assert to check document frequency
        assert(len(ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = self.compute_cider(ctest, crefs)
        return np.mean(np.array(score)), np.array(score)
