from models import ModelFactory
from utils import Beam
from tqdm import tqdm
import copy

import torch.nn.functional as F
import torch

def eval(args, test_loader, word_vectors):
    mf = ModelFactory(args, word_vectors)
    img_enc, sentence_dec, word_dec = mf.get_models()
    mf.load_models(img_enc, sentence_dec, word_dec)

    inv_vocab = {v: k for k, v in args.vocabulary.items()}

    for images, reports, num_sentences, word_lengths, prob in tqdm(test_loader):
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
                final_beam = Beam(args.beam_size)
                beam = Beam(args.beam_size)
                beam.add(([word_input.item()], wStates), 0)
                while make_words:
                    acc = []
                    for elem, beam_score in beam:
                        words, wState = elem
                        word_input = torch.tensor(words[-1]) \
                                          .unsqueeze(0) \
                                          .to(args.device)

                        if word_input.item() == args.vocabulary['<end>']:
                            final_beam.add(words, beam_score)
                            continue

                        scores, states = word_dec(img_features, img_avg_features, \
                                            topic, word_input, wState)
                        scores = F.log_softmax(scores, dim=1)
                        scores, indices = torch.topk(scores, args.beam_size, dim=1)
                        scores, indices = scores.squeeze(dim=0), indices.squeeze(dim=0)

                        for score, idx in zip(scores, indices):
                            w = copy.deepcopy(words) # 😐 ⏰⏱⏳
                            w.append(idx.item())
                            acc.append((score + beam_score, w, states))

                    beam = Beam(args.beam_size)
                    for score, words, states in acc:
                        beam.add((words, states), score)


                    make_words = len(final_beam) != args.beam_size
                    if make_words and word_idx >= word_lengths[0][sentence_idx]:
                        print('dang it, we cant stop making words up 🗣')
                        break
                    word_idx += 1

                sentence = final_beam.head() if len(final_beam) != 0 else beam.head()[0]
                pred_reports[0].append(sentence)
                generate = not (stop > 0.5).squeeze()[1].item()
                if generate and sentence_idx >= num_sentences[0] - 1:
                    print('dang it, we just dont know how to stop🛑✋🤔')
                    break
                sentence_idx += 1


            print("PRED:", [[inv_vocab[word_i] for word_i in sentence] for sentence in pred_reports[0]])
            print("TRUE:", [[inv_vocab[word_i.item()] for word_i in sentence] for sentence in reports[0]])


def greedy_eval(args, test_loader, word_vectors):
    mf = ModelFactory(args, word_vectors)
    img_enc, sentence_dec, word_dec = mf.get_models()

    inv_vocab = {v: k for k, v in args.vocabulary.items()}

    mf.load_models(img_enc, sentence_dec, word_dec)

    for images, reports, num_sentences, word_lengths, prob in tqdm(test_loader):
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
                    make_words = sentence[-1] != args.vocabulary['<end>']
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
