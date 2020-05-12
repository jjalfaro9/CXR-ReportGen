# Given the imgs, reports we will calculate the term frequencies of our dataset
# Which will be used later :D

import pickle

from collections import defaultdict

from typing import List, DefaultDict

from arg_parser import ArgParser
from data import CXRReports
from tqdm import tqdm
from cider_scorer import cook_refs

class TermFrequency:
    def __init__(self, args: 'argparse.Namespace'):
        self.document_frequency = defaultdict(float)
        self.data = CXRReports(args.data)
        self.ref_len = len(self.data)
        self.output_path = args.output_path

    def compute(self):
        for words in tqdm(self.data):
            if len(words) == 0:
                continue
            crefs = cook_refs(words)
            self._compute_word_freq(crefs)

    def _compute_word_freq(self, crefs: List[DefaultDict]):
        for ngram in set([ngram for ref in crefs for (ngram, count) in ref.items()]):
            self.document_frequency[ngram] += 1

    def save(self):
        data = {
            'document_frequency': self.document_frequency,
            'ref_len': self.ref_len
        }
        with open(self.output_path, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    parser = ArgParser()
    tf = TermFrequency(parser.parse_args())
    tf.compute()
    tf.save()
    
