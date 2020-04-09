'''Find the maximum sequence length across all captions. This is needed for
padding captions when input to the LSTM.
'''

import nltk
import pandas as pd
import pickle

from pathlib import Path
from tqdm import tqdm

from build_vocab import Vocabulary  # noqa: F401


def main():
    csv = pd.read_csv('p10.csv')
    lenmax = 0
#   captionmax = 0
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    for i, image in enumerate(tqdm(csv.path)):
        caption = Path(image).parent.as_posix() + '.txt'
        caption = open(caption).read()
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        if len(caption) > lenmax:
            lenmax = len(caption)
#           captionmax = caption
    print(f'Length of longest caption is {lenmax} tokens')
#   for idx in captionmax:
#       print(vocab.idx2word[idx], end=' ')
#   print()


if __name__ == '__main__':
    main()
