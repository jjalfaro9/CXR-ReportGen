import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LS
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
from torch.nn.utils.rnn import pack_padded_sequence
from models import ImageEncoder, SentenceDecoder, WordDecoder
from typing import *
import GPUtil
import time

def save_models(args, encoder, sentence_decoder, word_decoder, epoch, optimizer, loss):
    path = "save/"
    torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'sentence_decoder_state_dict': sentence_decoder.state_dict(),
            'word_decoder_state_dict': word_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, path + args.model_name+".pth")

def train(train_params, args, train_loader, val_loader):
    img_enc = ImageEncoder(args.embedd_size, args.hidden_size, args.img_size)
    sentence_dec = SentenceDecoder(args.vocab_size, args.hidden_size)
    word_dec = WordDecoder(args.vocab_size, args.hidden_size, args.img_feature_size)
    if args.parallel:
        img_enc = nn.DataParallel(img_enc, device_ids=args.gpus)
        sentence_dec = nn.DataParallel(sentence_dec, device_ids=args.gpus)
        word_dec = nn.DataParallel(word_dec, device_ids=args.gpus)


    img_enc.to(args.device)
    sentence_dec.to(args.device)
    word_dec.to(args.device)

    # DETAILS BELOW MATCHING PAPER
    if args.parallel:
        params = list(img_enc.module.affine_a.parameters()) \
                + list(img_enc.module.affine_b.parameters())
    else:
        params = list(img_enc.affine_a.parameters()) \
                + list(img_enc.affine_b.parameters())
    params = params + list (sentence_dec.parameters()) \
            + list(word_dec.parameters())

    optimizer = torch.optim.Adam(params, lr=train_params['lr'])
    scheduler = LS.MultiStepLR(optimizer, milestones=[16, 32, 48, 64], gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_encoder = None
    full_patience = 10
    patience = full_patience
    batch_size = train_params['batch_size']
    log_interval = int(len(train_loader) * 0.5)
    val_interval = int(len(train_loader))
    print('log_interval:', log_interval, 'val_interval:', val_interval, 'len of train', len(train_loader))

    start = time.time()
    for epoch in range(train_params['epochs']):
        print('== Epoch:', epoch)
        epoch_loss = 0
        train_loss = 0
        # TO-DO: Match sure to match DataLoader
        for batch_idx, (images, reports, num_sentences, word_lengths, prob) in enumerate(train_loader):

            print("BATCH STATUS:", batch_idx, (batch_idx+1)/len(train_loader), time.time()-start)
            start = time.time()

            # print("Start of batch:")
            # GPUtil.showUtilization()

            img_enc.train()
            sentence_dec.train()
            word_dec.train()
            if len(images) == 0:
                continue

            images = images.to(args.device)
            reports = reports.to(args.device)

            # print("Input size [img] [rep]", images.size(), reports.size(), np.prod(reports.shape))

            img_features, img_avg_features = img_enc(images)
            sentence_states = None
            sentence_loss = 0
            word_loss = 0
            # print('lets look at where these tensors live!', img_features.device, img_avg_features.device)
            # del images


            for sentence_idx in range(reports.shape[1]):
                stop_signal, topic_vec, sentence_states = sentence_dec(img_features, sentence_states)
                sentence_loss += criterion(stop_signal.squeeze(1), prob[:, sentence_idx].long().to(args.device)).long().sum()
                # TODO: do we need a sentence loss criterion???
                for word_idx in range(1, reports.shape[2] - 1):
                    scores = word_dec(img_features, img_avg_features, topic_vec, reports[:, sentence_idx, :word_idx])
                    golden = reports[:, sentence_idx, word_idx].to(args.device)
                    report_mask = (golden > 1).view(-1,).float()
                    # TODO: ensure report mask is correct. might need to update this mask
                    t_loss = criterion(scores, golden)
                    t_loss = t_loss * report_mask
                    word_loss += t_loss.sum()
                    # del golden
                # del stop_signal, topic_vec, scores #, atten_weights, beta

            loss = word_loss + sentence_loss
            optimizer.zero_grad()

            # print("Before loss")
            # GPUtil.showUtilization()

            loss.mean().backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()

            # print("before del:")
            # GPUtil.showUtilization()
            # del img_features, img_avg_features, sentence_states, loss

            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            # if batch_idx % log_interval == 0:
            #     idx = epoch * int(len(train_loader.dataset) / batch_size) + batch_idx
            #     writer.add_scalar('loss', train_loss.item(), idx)

        save_models(args, img_enc, sentence_dec, word_dec, epoch, optimizer, train_loss)
        epoch_loss = train_loss
        print(f"epoch loss: {epoch_loss}")
        scheduler.step()

    print('Finished: Best L1 loss on val:{}'.format(best_loss))
