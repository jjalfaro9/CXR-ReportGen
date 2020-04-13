import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor

from train import train
from data import CXRDataset, collate_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # DETAILS BELOW MATCHING PAPER
    parser.add_argument("--epochs", type=int, default=64,
                        help="Number of epochs to train")
    parser.add_argument("--model_name", type=str, default="model",
                        help="Name of model")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning Rate")
    parser.add_argument( '--embedd_size', type=int, default=256,
                        help='dimension of word embedding vectors, also dimension of v_g' )
    parser.add_argument( '--hidden_size', type=int, default=512,
                         help='dimension of lstm hidden states' )

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print(f"device: {args.device}")

    train_params = {
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': 8,
        'validate': True
    }

    print(f"ARGUMENTS: {args}\n")
    print(f"TRAIN PARAMS: {train_params}\n")

    train_dataset = CXRDataset('../data/', 'Train')
    val_dataset = CXRDataset('../data/', 'Validation')
    args.vocab_size = len(train_dataset.vocabulary)

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'],
                                shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'],
                                shuffle=True, collate_fn=collate_fn)


    train(train_params, args, train_loader, val_loader)
