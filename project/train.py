import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LS
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor

from encoders import Encoder, Decoder
from loss import cross_entropy_loss

def save_models(args, encoder, decoder, epoch, optimizer, loss):
    path = "save/"
    torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, path + args.encoder_name+".pth")

def train(train_params, args, train_loader, val_loader):
    encoder = Encoder()
    decoder = Decoder()
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)

    # DETAILS BELOW MATCHING PAPER
    optimizer = torch.optim.Adam(list(encoder.parameters(), decoder.parameters()), lr=train_params['lr'])
    scheduler = LS.MultiStepLR(optimizer, milestones=[16, 32, 48, 64], gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_encoder = None
    full_patience = 10
    patience = full_patience
    batch_size = train_params['batch_size']
    writer = SummaryWriter('log/{}'.format(args.encoder_name))
    log_interval = int(len(train_loader) / batch_size * 0.5)
    val_interval = int(len(train_loader) / batch_size)
    print('log_interval:', log_interval, 'val_interval:', val_interval)

    for epoch in range(train_params['epochs']):
        print('== Epoch:', epoch)
        epoch_loss = 0
        # TO-DO: Match sure to match DataLoader
        for batch_idx, (img, description) in enumerate(train_loader):
            img = img.to(args.device)

            enc_output = encoder(img)
            dec_output = decoder(enc_output)
            loss = criterion(dec_output, description)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss


            if batch_idx % log_interval == 0:
                idx = epoch * int(len(train_loader.dataset) / batch_size) + batch_idx
                writer.add_scalar('loss', loss.item(), idx)

            if batch_idx % val_interval == 0 and train_params['validate']:
                encoder.eval()
                val_loss = 0
                for batch_idx, (img, description) in enumerate(val_loader):
                    img = img.to(args.device)
                    enc_output = encoder(img)
                    dec_output = decoder(enc_output)

                    val_loss += criterion(dec_output, description).item()
                writer.add_scalar('val_loss', val_loss / len(val_loader), idx)
                writer.flush()

                if best_loss > val_loss:
                    best_loss = val_loss
                    best_encoder = copy.deepcopy(encoder)
                    best_decoder = copy.deepcopy(decoder)
                    save_models(args,
                        encoder,
                        decoder
                        epoch,
                        optimizer,
                        loss)
                    print('Improved: current best_loss on val:{}'.format(best_loss))
                    patience = full_patience
                else:
                    patience -= 1
                    print('patience', patience)
                    if patience == 0:
                        save_models(args,
                            encoder,
                            decoder,
                            epoch,
                            optimizer,
                            loss)
                        print('Early Stopped: Best L1 loss on val:{}'.format(best_loss))
                        writer.close()
                        return
            encoder.train()
        print(f"epoch loss: {epoch_loss}")
        writer.add_scalar('epoch loss', epoch_loss, epoch)
        scheduler.step()

    print('Finished: Best L1 loss on val:{}'.format(best_loss))
    writer.close()