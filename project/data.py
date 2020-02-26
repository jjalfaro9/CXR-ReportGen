import json
from pathlib import Path
from skimage.io import imread
from PIL import Image
import numpy as np
import os
import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor


class CXRDataset(Dataset):
    def __init__(self, dataset_path, split, transform=ToTensor()):
        self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.files[idx]

        # TO-DO: ACTUALLY FIGURE OUT HOW TO READ THE FILES

        return data