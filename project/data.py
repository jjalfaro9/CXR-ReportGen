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
    def __init__(self, split, transform=[Resize((256, 256)), ToTensor()], use_sample=False):
        self.files = []
        self.transform = transform
        self.use_sample = use_sample

        if self.use_sample:
            self.sample = '../png_files_sample/'
            self.files = []
            for file in os.listdir(self.sample + 'img'):
                self.files.append(file[:-4])

            self.vocabulary = pickle.load(open('sample_idxr-obj', 'rb'))
        else:
            self.data_path = '../data/'+split.lower()
            self.images = []
            for line in open(self.data_path+'_images.txt'):
                self.images.append(line.strip())

            self.reports = []
            for line in open(self.data_path+'_reports.txt'):
                self.reports.append(line.strip())

            self.vocabulary = pickle.load(open('full_idxr-obj', 'rb'))
        self.s_max = 8
        self.n_max = 18

    def __len__(self):
        if self.use_sample:
            return len(self.files)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        image_to_tensor = transforms.Compose(self.transform)
        if self.use_sample:
            img_path = self.sample + 'img/' + self.files[idx] +'.png'
            report_path = self.sample + 'label/' + self.files[idx] +'.txt'

            img = image_to_tensor(Image.open(img_path).convert('RGB'))
        else:
            img_path = self.images[idx]
            report_path = self.reports[idx]

            with pydicom.read_file(img_path) as ds:
                # convert to png from https://github.com/pydicom/pydicom/issues/352#issuecomment-406767850
                # Convert to float to avoid overflow or underflow losses.
                image_2d = ds.pixel_array.astype(float)
                # Rescaling grey scale between 0-255
                image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
                # Convert to uint
                image_2d_scaled = np.uint8(image_2d_scaled)

            pilImg = Image.fromarray(image_2d_scaled)
            img = image_to_tensor(pilImg)
            img = torch.repeat(3, 1, 1) # per @rjrock

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

        num_sentences = len(target)
        return (img, target, num_sentences, longest_sentence_length, img_path)

def collate_fn(data):
    pre_images, pre_captions, num_sentences, longest_sentence_length, pre_image_paths = zip(*data)
    # remove empty image-caption pairs
    images = []
    captions = []
    image_paths = []
    for i in range(len(pre_captions)):
        cap = pre_captions[i]
        if len(cap) > 0:
            images.append(pre_images[i])
            captions.append(pre_captions[i])
            image_paths.append(pre_image_paths[i])
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

    return images, targets, num_sentences, word_lengths, prob, image_paths


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
