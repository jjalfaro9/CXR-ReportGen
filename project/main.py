import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor

from train import train
from models import *
from data import CXRDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # DETAILS BELOW MATCHING PAPER
    parser.add_argument("--epochs", type=int, default=64,
                        help="Number of epochs to train")
    parser.add_argument("--model_name", type=str, default="model",
                        help="Name of model")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning Rate")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print(f"device: {args.device}")

    train_params = {
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': 64,
        'validate': True
    }

    print(f"ARGUMENTS: {args}\n")
    print(f"TRAIN PARAMS: {train_params}\n")

    train_dataset = MSRCDataset('../data/', 'Train')
    val_dataset = MSRCDataset('../data/', 'Validation')

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=True)


    train(train_params, args, train_loader, val_loader)