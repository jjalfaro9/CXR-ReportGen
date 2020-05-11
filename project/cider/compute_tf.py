# Given the imgs, reports we will calculate the term frequencies of our dataset
# Which will be used later :D

# This will compute the needed dictionaries and pickle them!
# You can do it here in this directory!
import pickle

from torch.utils.data import DataLoader
from typing import List, Dict

from arg_parser import ArgParser
from data import CXRReports
from tqdm import tqdm
from cider_scorer import cook_refs

class TermFrequency:
    def __init__(self, args: 'argparse.Namespace'):
        self.document_frequency = defaultdict(float)
        data = CXRReports(args.data)
        self.loader = DataLoader(data, batch_size=1, shuffle=False)
        self.ref_len = len(self.loader)
        self.output_path = args.output_path

    def compute(self):
        for words in tqdm(self.loader):
            crefs = cook_refs(words)
            self._compute_word_freq(crefs)

    def _compute_word_freq(self, crefs: List[Dict]):
        for refs in crefs:
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
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
    
