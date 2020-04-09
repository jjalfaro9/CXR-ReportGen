'''build_vocab.py'''

import nltk
import pandas as pd
import pickle

from collections import Counter
from pathlib import Path
from tqdm import tqdm


cwd = Path(__file__).resolve().parent
data = cwd/'data'


class Vocabulary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(threshold):
    csv_file = data/'p10.csv'
    counter = Counter()
    csv = pd.read_csv(csv_file)
    for i, path in enumerate(tqdm(csv.path)):
        f = data/(Path(path).parent.with_suffix('.txt'))
        caption = open(f.as_posix()).read()
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main():
    vocab = build_vocab(threshold=4)
    vocab_path = f'{data}/vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f'Total vocabulary size: {len(vocab)}')
    print(f'Saved the vocabulary wrapper to {vocab_path}')


if __name__ == '__main__':
    main()
