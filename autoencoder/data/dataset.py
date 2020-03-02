'''Return a dataset that generates images (and reports?) from a csv file.'''

import cv2
import numpy as np
import os
import pydicom
import torch

from pathlib import Path
from torch.utils.data import Dataset


class ToTensor():
    def __call__(self, img):
        '''Preprocess the image for DenseNet-121.'''
        # TODO: Is it better to do this using a pytorch tensor on the
        # GPU?
        img = cv2.resize(img, dsize=(256, 256))
        scaled = (img/np.max(img)).astype(np.float)
        rgb = np.repeat(scaled[..., np.newaxis], 3, -1)
        return rgb.transpose((2, 0, 1))


class CSVDataset(Dataset):
    '''This assumes that the csv contains filenames relative to the
    directory this file is located.
    '''
    def __init__(self, csv_file):
        self.cwd = Path(os.path.dirname(__file__))
        csv_file = self.cwd/csv_file
        self.filenames = open(csv_file).read().splitlines()
        self.transform = ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = (self.cwd/self.filenames[idx]).as_posix()
        img = pydicom.dcmread(filename).pixel_array
        return self.transform(img)
