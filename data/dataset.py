'''Return a dataset that generates images and captions from a csv
file.'''

import csv
import cv2
# import json
import nltk
import numpy as np
import os
import torch

import settings

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class ToTensor():
    def __call__(self, img):
        '''Preprocess the image for DenseNet-121.'''
        img = cv2.resize(img, dsize=(256, 256))
        img = (img/np.max(img)).astype(np.float)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(img)
        # How necessary is normalizing? It makes the data not be between [0,
        # 1].
#       normalize = transforms.Compose([
#           transforms.ToTensor(),
#           transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.225, 0.225])
#       ])
#       return normalize(rgb)


class CSVDataset(Dataset):
    '''This assumes that the csv contains filenames relative to the
    directory this file is located.
    '''
    def __init__(self, csv_file, vocab):
        self.cwd = Path(os.path.dirname(__file__))
        csv_file = self.cwd/csv_file
        reader = csv.reader(open(csv_file), delimiter=',')
        data = [line for line in reader]
        self.images = [pair[0] for pair in data]
        self.captions = [pair[1] for pair in data]
        self.vocab = vocab
        self.transform = ToTensor()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Need to include the view as well. Accoriding to the paper the view
        # can be in three orientations
        #   AP: anteroposterior
        #   PA: posteroanterior
        #   LL: lateral
        # Concat view with image? And untangle at encoder?
        path = (self.cwd/self.images[idx])
#       json_file = open(path.with_suffix('.json').as_posix())
#       view = json.load(json_file)['view']
        png = path.with_suffix('.png').as_posix()
        image = cv2.imread(png)
        image = self.transform(image)
        caption = open(self.cwd/self.captions[idx]).read()
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return image.float(), target


def collate_fn(data):
    '''Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.

    Taken from pytorch tutorials.
    '''
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(csv_file, vocab):
    dataset = CSVDataset(csv_file, vocab)
    kwargs = {'num_workers': 1, 'pin_memory': True} if settings.cuda else {}
    loader = DataLoader(dataset,
                        batch_size=settings.batch_size,
                        collate_fn=collate_fn,
                        shuffle=True,
                        **kwargs)
    return loader
