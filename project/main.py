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
    parser.add_argument('--batch_size', type=int, default=32,
                         help='batch size for data')
    parser.add_argument('--data_workers', type=int, default=0)
    parser.add_argument('--pin_mem', type=bool, default=False)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    args.parallel = torch.cuda.device_count() > 1
    args.gpus = range(torch.cuda.device_count())
    print("device:", args.device, "parallel:", args.parallel, "gpus:", args.gpus)

    train_params = {
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'validate': True
    }

    print("ARGUMENTS:", args, "\n")
    print("TRAIN PARAMS:", train_params, "\n")

    train_dataset = CXRDataset('../data/', 'Train')
    val_dataset = CXRDataset('../data/', 'Validation')
    args.vocab_size = len(train_dataset.vocabulary)

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'],
                                shuffle=True, collate_fn=collate_fn, num_workers=args.data_workers, pin_memory=args.pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'],
                                shuffle=True, collate_fn=collate_fn)


    train(train_params, args, train_loader, val_loader)
