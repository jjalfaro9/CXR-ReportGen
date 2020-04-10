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
    def __init__(self, dataset_path, split, transform=[Resize((394, 300)), ToTensor()]):
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

        target = []
        longest_sentence_length = 0

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

            for i, sentence in enumerate(sentences):
                sentence = sentence.lower().replace('.', '').replace(',', '').split()
                if len(sentence) == 0: # or len(sentence) > self.n_max:
                    continue
                tokens = list()
                tokens.append(self.vocabulary['<start>'])
                tokens.extend([self.vocabulary[token] for token in sentence])
                tokens.append(self.vocabulary['<end>'])
                if longest_sentence_length < len(tokens):
                    longest_sentence_length = len(tokens)
                target.append(tokens)

            # return (img, target)
        except ValueError:
            pass
            # if 'FINDINGS' in file_read:
            #     print(self.files[idx])
            #     with open (report_path, "r") as r_file:
            #         print("FILE \n", file_read, "\n")

            #     print("\n ---------------------------------------- \n")

            #     raise ValueError

        num_sentences = len(target)
        return (img, target, num_sentences, longest_sentence_length)

def collate_fn(data):
    pre_images, pre_captions, num_sentences, longest_sentence_length = zip(*data)
    
    # remove empty image-caption pairs
    images = []
    captions = []
    for i in range(len(pre_captions)):
        cap = pre_captions[i]
        if len(cap) > 0:
            images.append(pre_images[i])
            captions.append(pre_captions[i])

    try:
        images = torch.stack(images, 0)
    except RuntimeError: #if the batch ends up being fully corrupt
        images = torch.tensor(images)

    max_sentence_num = max(num_sentences)
    max_word_num = max(longest_sentence_length)

    targets = np.zeros((len(captions), max_sentence_num, max_word_num))
    prob = np.zeros((len(captions), max_sentence_num))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i, j] = 1

    targets = torch.Tensor(targets).long()
    prob = torch.Tensor(prob)

    return images, targets, prob
        

# train_dataset = CXRDataset('../data/', 'Train')
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# skip = 0
# for batch_idx, (images, target, prob) in enumerate(train_loader):
#     print("TARGET", target.shape)
#     try:
#         if target.shape[0] == 0:
#             skip += 1
#     except IndexError:
#         print("\nDESCRIPTION:", description)

# print(skip)
# print(len(train_dataset))

