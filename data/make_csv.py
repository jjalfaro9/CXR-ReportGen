'''Create train and test files to be parsed through dataset.py and used
to train and evaluate the autoencoder.
'''

import pandas as pd

from pathlib import Path
from tqdm import tqdm


def main():
    csv = pd.read_csv('p10.csv')
    images = []
    captions = []
    for i, image in enumerate(tqdm(csv.path)):
        images.append(image)
        caption = Path(image).parent.with_suffix('.txt').as_posix()
        captions.append(caption)
    split = 2000
    stitch = lambda x: ','.join([*x])
    print('\n'.join(map(stitch, zip(images[:-split], captions[:-split]))),
          file=open('train.csv', 'w'))
    print('\n'.join(map(stitch, zip(images[-split:], captions[-split:]))),
          file=open('test.csv', 'w'))


if __name__ == '__main__':
    main()
