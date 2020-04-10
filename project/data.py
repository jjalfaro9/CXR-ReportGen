import json
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

from tqdm import tqdm
import re
import pickle


class CXRDataset(Dataset):
    def __init__(self, dataset_path, split, transform=[ToTensor()]):
        self.files = []
        self.transform = transform

        # sample dataset for testing
        self.sample = '../png_files_sample/'
        self.files = []
        for file in os.listdir(self.sample + 'img'):
            self.files.append(file[:-4])

        self.vocabulary = pickle.load(open('sample_idxr-obj', 'rb'))

        # self.p10 = '/data/mimic-cxr/files/p10/'
        # self.p11 = '/data/mimic-cxr/files/p11/'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.sample + 'img/' + self.files[idx] +'.png'
        report_path = self.sample + 'label/' + self.files[idx] +'.txt'
        
        image_to_tensor = transforms.Compose(self.transform)
        img = image_to_tensor(Image.open(img_path))

        try:
            with open (report_path, "r") as r_file:
                file_read = r_file.read()
                report = re.split("[\n:]", file_read)
                for i in range(len(report)):
                    report[i] = report[i].strip().lower()

            try:
                index = report.index('findings')
            except ValueError:
                index = report.index('findings and impression')
            try:
                index2 = report.index('impression')
            except ValueError:
                index2 = len(report)

            sentences = ' '.join(report[index+2:index2]).split('. ')

            target = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.lower().replace('.', '').replace(',', '').split()
                if len(sentence) == 0: # or len(sentence) > self.n_max:
                    continue
                tokens = list()
                tokens.append(self.vocabulary['<start>'])
                tokens.extend([self.vocabulary[token] for token in sentence])
                tokens.append(self.vocabulary['<end>'])
                # if word_num < len(tokens):
                #     word_num = len(tokens)
                target.append(tokens)

            return (img, target)
        except ValueError:
            # if 'FINDINGS' in file_read:
            #     print(self.files[idx])
            #     with open (report_path, "r") as r_file:
            #         print("FILE \n", file_read, "\n")

            #     print("\n ---------------------------------------- \n")

            #     raise ValueError

            return (img, "skip")

        

train_dataset = CXRDataset('../data/', 'Train')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

skip = 0
for batch_idx, (_, description) in tqdm(enumerate(train_loader)):
    try:
        if description[0] == 'skip':
            skip += 1
    except IndexError:
        print("\nDESCRIPTION:", description)

print(skip)
print(len(train_dataset))

