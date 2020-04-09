'''train_caption.py'''

import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn

import data
import model
import settings

from pathlib import Path
from torch.nn.utils.rnn import pack_padded_sequence

from build_vocab import Vocabulary  # noqa: F401


cwd = Path(__file__).resolve().parent


def run(args):
    device = settings.device
    train_csv = 'train.csv'
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    train_loader = data.get_loader(train_csv, vocab)

    encoder = model.Encoder(dim=256).to(device)
    decoder = model.Decoder(embed_size=256, hidden_size=512,
                            vocab_size=len(vocab), num_layers=1).to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(encoder.projection.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    total_step = len(train_loader)

    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets =\
                pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_step == 0:
                print(f'Epoch [{epoch}/{args.num_epochs}],'
                      f' Step [{i}/{total_step}],'
                      f' Loss: {loss.item():.4f},'
                      f' Perplexity: {np.exp(loss.item()):5.4f}')

            if (i+1) % args.save_step == 0:
                decoder_save = (args.model_path
                                /f'decoder-{epoch+1}-{i+1}.ckpt').as_posix()
                encoder_save = (args.model_path
                                /f'encoder-{epoch+1}-{i+1}.ckpt').as_posix()
                torch.save(decoder.state_dict(), decoder_save)
                torch.save(encoder.state_dict(), encoder_save)


def parse_args():
    parser = argparse.ArgumentParser(description='Captioner')
    parser.add_argument('--model_path', type=Path,
                        default=f'{cwd}/model/saved',
                        help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str,
                        default=f'{cwd}/data/vocab.pkl',
                        help='Location of one-hot vocabulary')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')
    args = parser.parse_args()
    args.model_path.mkdir(exist_ok=True, parents=True)
    return args


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
