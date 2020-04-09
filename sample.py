'''sample.py'''

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

import settings

from PIL import Image
from models import Encoder, Decoder
from torchvision import transforms

from build_vocab import Vocabulary  # noqa: F401


def load_image(image_path, transform=None):
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(256, 256))
    img = (img/np.max(img)).astype(np.float)

    if transform is not None:
        img = transform(img).unsqueeze(0)

    return img.float()


def main(args):
    device = settings.device
#   transform = transforms.Compose([
#       transforms.ToTensor(),
#       transforms.Normalize((0.485, 0.456, 0.406),
#                            (0.229, 0.224, 0.225))])

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder = Encoder(args.embed_size).eval()
    decoder = Decoder(args.embed_size, args.hidden_size, len(vocab),
                      args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    print(sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str,
                        default='model/saved/encoder-1-1000.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str,
                        default='model/saved/decoder-1-1000.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
