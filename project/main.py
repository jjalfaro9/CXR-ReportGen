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
    parser.add_argument('--data_workers', type=int, default=8)
    parser.add_argument('--pin_mem', type=bool, default=False)
    parser.add_argument('--img_size', type=int, default=128)
    parallel_parser = parser.add_mutually_exclusive_group(required=False)
    parallel_parser.add_argument('--parallel', dest='parallel', action='store_true')
    parallel_parser.add_argument('--no-parallel', dest='parallel', action='store_false')
    parser.set_defaults(parallel=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    args.parallel = torch.cuda.device_count() > 1 and args.parallel
    args.gpus = range(torch.cuda.device_count())

    train_params = {
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'validate': True
    }

    train_dataset = CXRDataset('../data/', 'Train', transform=[Resize((args.img_size, args.img_size)), ToTensor()])
    val_dataset = CXRDataset('../data/', 'Validation', transform=[Resize((args.img_size, args.img_size)), ToTensor()])
    args.vocab_size = len(train_dataset.vocabulary)
    args.img_feature_size = (args.img_size // 32) ** 2

    print("ARGUMENTS:", args, "\n")
    print("TRAIN PARAMS:", train_params, "\n")

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'],
                                shuffle=False, collate_fn=collate_fn, num_workers=args.data_workers, pin_memory=args.pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'],
                                shuffle=False, collate_fn=collate_fn)


    train(train_params, args, train_loader, val_loader)
