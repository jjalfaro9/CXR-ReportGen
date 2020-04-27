import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
import pydicom
import io

class CXRDataset(Dataset):
    def __init__(self, split, transform=[Resize((256, 256)), ToTensor()]):
        self.files = []
        self.transform = transform

        # sample dataset for testing
        self.data_path = '../data/'

        self.images = []
        for line in open(self.data_path+'all_images.txt'):
            self.images.append(line.strip())

        self.reports = []
        for line in open(self.data_path+'all_reports.txt'):
            self.reports.append(line.strip())

        self.vocabulary = pickle.load(open('full_idxr-obj', 'rb'))

        self.s_max = 6
        self.n_max = 13

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        report_path = self.reports[idx]

        # ds = pydicom.dcmread(img_path)
        ds = pydicom.read_file(img_path).pixel_array
        # cmap_reversed = matplotlib.cm.get_cmap('binary_r')
        # plt.imshow(ds.pixel_array, cmap=cmap_reversed)
        # plt.gca().axes.get_yaxis().set_visible(False)
        # plt.gca().axes.get_xaxis().set_visible(False)

        # buf = io.BytesIO()
        # plt.savefig(buf, bbox_inches='tight')
        # buf.seek(0)

        # TODO: Don't convert to RGB
        image_to_tensor = transforms.Compose(self.transform)
        # img = image_to_tensor(Image.open(buf).convert('RGB'))
        img = image_to_tensor(Image.fromarray(ds).convert('RGB'))

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
                if len(target) >= self.s_max:
                    break
                sentence = sentence.lower().replace('.', '').replace(',', '').split()
                if len(sentence) == 0 or len(sentence) > self.n_max:
                    continue
                tokens = list()
                tokens.append(self.vocabulary['<start>'])
                tokens.extend([self.vocabulary[token] for token in sentence])
                tokens.append(self.vocabulary['<end>'])
                if longest_sentence_length < len(tokens):
                    longest_sentence_length = len(tokens)
                target.append(tokens)

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
    num_sentences = [len(cap) for cap in captions]

    try:
        images = torch.stack(images, 0)
    except RuntimeError: #if the batch ends up being fully corrupt
        images = torch.tensor(images)
    try:
        max_sentence_num = max(num_sentences)
    except ValueError:
        max_sentence_num = 0
    max_word_num = max(longest_sentence_length)

    targets = np.zeros((len(captions), max_sentence_num, max_word_num))
    word_lengths = np.zeros((len(captions), max_sentence_num))
    prob = np.zeros((len(captions), max_sentence_num))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            word_lengths[i, j] = len(sentence)
            prob[i, j] = 1

    targets = torch.Tensor(targets).long()
    prob = torch.Tensor(prob)

    return images, targets, num_sentences, word_lengths, prob


def main():
    train_dataset = CXRDataset('Train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    skip = 0
    for batch_idx, (images, targets, num_sentences, word_lengths, prob) in enumerate(train_loader):
        print("BATCH STATUS:", (batch_idx+1)/len(train_loader))
        print("TARGET", images.shape, targets.shape)
        try:
            if targets.shape[0] == 0:
                skip += 1
        except IndexError:
            print("\nDESCRIPTION:", description)

    print(skip)
    print(len(train_dataset))

# main()

