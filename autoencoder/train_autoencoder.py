'''Train autoencoder.'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import data
import settings

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models.autoencoder import Autoencoder

cwd = Path(os.path.dirname(__file__))
results = cwd/'results'
viz = results/'autoencoder'
viz.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Autoencoder')
    parser.add_argument('--savefile', type=str,
                        default=f'{cwd}/models/autoencoder.pt',
                        help='Location to save autoencoder parameters.')
    # TODO: Add option to load from provided .pt file
    args = parser.parse_args()
    return args


def train_on_data(vae, epoch, train_loader):
    vae.train()
    data = train_loader.__iter__()
    length = len(train_loader)
    average_loss = 0
    with tqdm(total=length) as pbar:
        for i in range(1, length+1):
            loss = vae.step(next(data).float())
            average_loss = (i * average_loss + loss.item()) / (i + 1)
            pbar.set_description(f'Average batch loss {average_loss:.3f}')
            pbar.update(1)


def test_on_data(vae, epoch, test_loader):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data = data.float()
            data = data.to(settings.device)
            recon_batch = vae.forward(data)
            test_loss += vae.loss_function(recon_batch, data).item()
    return test_loss / len(test_loader)


def plot_autoencoding(vae, dataset, epoch):
    '''Visualize the autoencoding.'''
    num_comparisons = 8
    indices = np.random.randint(len(dataset), size=num_comparisons)
    images = np.array([dataset[i] for i in indices])
    tensor = torch.from_numpy(np.array(images)).to(settings.device)
    reconstruction_tensor = vae(tensor.float())
    reconstruction = reconstruction_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(4*num_comparisons, num_comparisons))
    for i in range(num_comparisons):
        plt.subplot(2, num_comparisons, i+1)
        plt.imshow(images[i].transpose((1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, num_comparisons, num_comparisons+i+1)
        plt.imshow(reconstruction[i].transpose((1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{viz}/reconstruction_{epoch}.png')


def run(model, savefile):
    train_csv = os.path.join('train.csv')
    test_csv  = os.path.join('test.csv')
    train_dataset = data.CSVDataset(train_csv)
    test_dataset = data.CSVDataset(test_csv)

    kwargs = {'num_workers': 1, 'pin_memory': True} if settings.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size,
                             shuffle=True, **kwargs)
    for epoch in range(settings.num_epochs):
        train_on_data(model, epoch, train_loader)
        loss = test_on_data(model, epoch, test_loader)
        print(f'====> Test set average batch loss: {loss:.4f}')
        torch.save(model.state_dict(), f'{savefile}')  # Save after every epoch
        with torch.no_grad():
            plot_autoencoding(model, test_dataset, epoch)


def main():
    args = parse_args()
    savefile = Path(args.savefile)
    input_size = (3, 256, 256)
    autoencoder = Autoencoder(input_size=input_size, dim=1000)
    autoencoder = autoencoder.to(settings.device)
    run(autoencoder, savefile)


if __name__ == '__main__':
    main()
