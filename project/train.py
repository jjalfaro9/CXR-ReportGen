import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LS
import tqdm

from torch.utils.tensorboard import SummaryWriter
from models import ImageEncoder, SentenceDecoder, WordDecoder

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

def get_models(args, word_vectors):
    img_enc = ImageEncoder(args.embedd_size, args.hidden_size, args.img_size)
    sentence_dec = SentenceDecoder(args.hidden_size)
    word_dec = WordDecoder(args.vocab_size, args.hidden_size, args.img_feature_size, word_vectors)
    if args.parallel:
        img_enc = nn.DataParallel(img_enc, device_ids=args.gpus)
        sentence_dec = nn.DataParallel(sentence_dec, device_ids=args.gpus)
        word_dec = nn.DataParallel(word_dec, device_ids=args.gpus)


    img_enc.to(args.device)
    sentence_dec.to(args.device)
    word_dec.to(args.device)
    return img_enc, sentence_dec, word_dec

def train(train_params, args, train_loader, val_loader, word_vectors):
    img_enc, sentence_dec, word_dec = get_models(args, word_vectors)

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
    start_epochs = 0

    if args.continue_training:
        model_dict = torch.load('save/{model}.pth'.format(model=args.model_name), map_location=args.device)
        img_enc.load_state_dict(model_dict['encoder_state_dict'])
        sentence_dec.load_state_dict(model_dict['sentence_decoder_state_dict'])
        word_dec.load_state_dict(model_dict['word_decoder_state_dict'])
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        start_epochs = model_dict['epoch']

    best_loss = float('inf')
    writer = SummaryWriter('log/{}'.format(args.model_name))
    batch_size = args.batch_size
    log_interval = int(len(train_loader) * 0.5)
    val_interval = int(len(train_loader))

    for epoch in range(start_epochs, train_params['epochs']):
        print('=' * epoch,' Epoch:', epoch)
        epoch_loss = 0
        train_loss = 0
        for batch_idx, (images, reports, num_sentences, word_lengths, prob) in enumerate(tqdm.tqdm(train_loader)):
            img_enc.train()
            sentence_dec.train()
            word_dec.train()

            if len(images) == 0:
                continue

            images = images.to(args.device)
            reports = reports.to(args.device)

            img_features, img_avg_features = img_enc(images)

            sentence_states = None
            sentence_loss = 0
            word_loss = 0

            max_generation = max(num_sentences) - 1
            num_sentences = torch.tensor(num_sentences) \
                                 .to(args.device)
            generate = True
            sentence_idx = 0
            # the random value is so that we can stop teacher forcing half way through epochs
            p = epoch + torch.LongTensor(1).random_(0, train_params['epochs'] // 2).item()
            teach_enforce_ratio = args.teacher_forcing_const ** (p)
            curr_batch_size = len(num_sentences)

            while generate:
                stop, topic, sentence_states = sentence_dec(img_avg_features, sentence_states)

                enforce = teach_enforce_ratio > 1 - teach_enforce_ratio

                prev_out = torch.tensor(args.vocabulary['<start>']) \
                                  .expand(curr_batch_size) \
                                  .to(args.device)
                h_z = torch.zeros(curr_batch_size, args.hidden_size) \
                         .to(args.device)
                c_z = torch.zeros(curr_batch_size, args.hidden_size) \
                         .to(args.device)                  
                wStates = (h_z, c_z)
                for word_idx in range(1, reports.shape[2]):
                    golden_words = reports[:, sentence_idx, word_idx]
                    word_input = reports[:, sentence_idx, word_idx-1] if enforce else prev_out

                    scores, wStates = word_dec(img_features, img_avg_features, \
                                                topic, word_input, wStates)

                    prev_out = torch.argmax(scores, dim=1)

                    masky_mask = (golden_words >= 1).long()
                    golden_words = golden_words[masky_mask.nonzero()].squeeze(1)
                    scores = scores[masky_mask.nonzero()].squeeze(1)
                    if golden_words.nelement() > 0:
                        w_l = criterion(scores, golden_words)
                        word_loss += w_l


                sen_stop = (num_sentences < sentence_idx).long()
                s_l = criterion(stop, sen_stop)
                sentence_loss += s_l

                sentence_idx += 1
                generate = sentence_idx < max_generation

            loss = args.lambda_sent * sentence_loss + args.lambda_word * word_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()

            if batch_idx % log_interval == 0:
                idx = epoch * int(len(train_loader.dataset) / batch_size) + batch_idx
                writer.add_scalar('train_loss', train_loss.item(), idx)
                writer.add_scalar('word_loss', word_loss.item(), idx)
                writer.add_scalar('sen_loss', sentence_loss.item(), idx)

        save_models(args, img_enc, sentence_dec, word_dec, epoch, optimizer, train_loss)
        epoch_loss = train_loss / len(train_loader)
        print(f"epoch loss: {epoch_loss}")
        writer.add_scalar('epoch loss', epoch_loss, epoch)
        scheduler.step()

    writer.close()

def test(args, test_loader, word_vectors):
    img_enc, sentence_dec, word_dec = get_models(args, word_vectors)

    inv_vocab = {v: k for k, v in args.vocabulary.items()}

    model_dict = torch.load('save/{model}.pth'.format(model=args.model_name), map_location=args.device)
    img_enc.load_state_dict(model_dict['encoder_state_dict'])
    sentence_dec.load_state_dict(model_dict['sentence_decoder_state_dict'])
    word_dec.load_state_dict(model_dict['word_decoder_state_dict'])

    for batch_idx, (images, reports, num_sentences, word_lengths, prob) in enumerate(tqdm.tqdm(test_loader)):
        img_enc.eval()
        sentence_dec.eval()
        word_dec.eval()

        if len(images) == 0:
            continue

        images = images.to(args.device)
        reports = reports.to(args.device)
        with torch.no_grad():
            img_features, img_avg_features = img_enc(images)

            sentence_states = None

            pred_reports = [[]]
            s_max = 8
            n_max = 18
            generate = True
            sentence_idx = 0
            while generate:
                stop, topic, sentence_states = sentence_dec(img_avg_features, sentence_states)
                word_input = torch.tensor(args.vocabulary['<start>']) \
                                  .unsqueeze(0) \
                                  .to(args.device)
                sentence = [word_input.item()]
                h_z = torch.zeros(1, args.hidden_size) \
                         .to(args.device)
                c_z = torch.zeros(1, args.hidden_size) \
                         .to(args.device)
                wStates = (h_z, c_z)
                word_idx = 0
                make_words = True
                while make_words:
                    scores, wStates = word_dec(img_features, img_avg_features, \
                                        topic, word_input, wStates)
                    word_input = torch.argmax(scores, dim=1)
                    sentence.append(word_input.item())
                    make_words = sentence[-1] != args.vocabulary['end']
                    if make_words and word_idx >= word_lengths[0][sentence_idx]:
                        print('dang it, we cant stop making words up 🗣')
                        break
                    word_idx += 1
                pred_reports[0].append(sentence)
                generate = not (stop > 0.5).squeeze()[1].item()
                if generate and sentence_idx >= num_sentences[0] - 1:
                    print('dang it, we just dont know how to stop🛑✋🤔')
                    break
                sentence_idx += 1


            print("PRED:", [[inv_vocab[word_i] for word_i in sentence] for sentence in pred_reports[0]])
            print("TRUE:", [[inv_vocab[word_i.item()] for word_i in sentence] for sentence in reports[0]])
