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
from torchvision.transforms import Resize
from torch.nn.utils.rnn import pack_padded_sequence
from models import ImageEncoder, SentenceDecoder, WordDecoder
from typing import *

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
    img_enc = ImageEncoder(args.embedd_size, args.hidden_size)
    sentence_dec = SentenceDecoder(args.vocab_size, args.hidden_size)
    word_dec = WordDecoder(args.vocab_size, args.hidden_size)

    # DETAILS BELOW MATCHING PAPER
    params = list(img_enc.affine_a.parameters()) \
            + list(img_enc.affine_b.parameters()) \
            + list (sentence_dec.parameters()) \
            + list(word_dec.parameters())

    optimizer = torch.optim.Adam(params, lr=train_params['lr'])
    scheduler = LS.MultiStepLR(optimizer, milestones=[16, 32, 48, 64], gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    test([img_enc, sentence_dec, word_dec], train_loader, criterion, optimizer)

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
        train_loss = 0
        # TO-DO: Match sure to match DataLoader
        for batch_idx, (images, targets, num_sentences, word_lengths, prob) in enumerate(train_loader):
            image_enc.train()
            sentence_dec.train()
            word_dec.train()
            if len(images) == 0:
                continue

            img_features, img_avg_features = image_enc(images)
            sentence_states = None
            sentence_loss = 0
            word_loss = 0

            for sentence_idx in range(reports.shape[1]):
                stop_signal, topic_vec, sentence_states = sentence_dec(img_features, sentence_states)
                # TODO: do we need a sentence loss criterion???
                for word_idx in range(1, reports.shape[2] - 1):

                    scores, _ = word_dec(img_features, img_avg_features, topic_vec, reports[:, sentence_idx, :word_idx])
                    report_mask = (reports[:, sentence_index, word_index] > 1).view(-1,).float()
                    # TODO: ensure report mask is correct. might need to update this mask
                    t_loss = self.criterion(scores, report[:, sentence_index, word_index])
                    t_loss = t_loss * caption_mask
                    word_loss += t_loss.sum()
            loss = word_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]


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
