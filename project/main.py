import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor
from gensim.models import KeyedVectors

from train import train, test
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
    parser.add_argument('--batch_size', type=int, default=16,
                         help='batch size for data')
    parser.add_argument('--data_workers', type=int, default=8)
    parser.add_argument('--pin_mem', type=bool, default=False)
    parser.add_argument('--use_sample', type=bool, default=False)
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--word_vecs_path', type=str, default='glove256_vocab_full.kv', help='path to word vectors file')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--teacher_forcing_const', type=float, default=.99)
    parser.add_argument('--lambda_sent', type=float, default=5.0)
    parser.add_argument('--lambda_word', type=float, default=1.0)
    parallel_parser = parser.add_mutually_exclusive_group(required=False)
    parallel_parser.add_argument('--parallel', dest='parallel', action='store_true')
    parallel_parser.add_argument('--no-parallel', dest='parallel', action='store_false')
    parser.set_defaults(parallel=False)
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
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

    train_dataset = CXRDataset('train', transform=[Resize((args.img_size, args.img_size)), ToTensor()], use_sample=args.use_sample)
    test_dataset = CXRDataset('test', transform=[Resize((args.img_size, args.img_size)), ToTensor()], use_sample=args.use_sample)
    args.vocab_size = len(train_dataset.vocabulary)
    args.img_feature_size = (args.img_size // 32) ** 2
    word_vectors = KeyedVectors.load(args.word_vecs_path, mmap='r').vectors

    print("ARGUMENTS:", args, "\n")
    print("TRAIN PARAMS:", train_params, "\n")

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'],
                                shuffle=False, collate_fn=collate_fn, num_workers=args.data_workers, pin_memory=args.pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


    train(train_params, args, train_loader, test_loader, word_vectors)
    # test(train_params, args, test_loader, word_vectors)
