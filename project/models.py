import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import init

from att import AdaptiveAttention

class ImageEncoder(nn.Module):
    def __init__(self, embedd_size, hidden_size, img_size):
        super(ImageEncoder, self).__init__()
        densenet121 = models.densenet121(pretrained=True)
        self.d121 = nn.Sequential(
            *list(densenet121.features.children())[:-1]
        )
        self.d121.requires_grad = False

        self.avgpool = nn.AvgPool2d(img_size//32)
        self.dropout = nn.Dropout(0.5)

        self.affine_a = nn.Linear(1024, hidden_size)
        self.affine_b = nn.Linear(1024, embedd_size)
        # self.affine_c = nn.Linear(img_size*img_size, hidden_size)

        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.affine_a.weight, mode='fan_in')
        init.kaiming_uniform_(self.affine_b.weight, mode='fan_in')
        # init.kaiming_uniform_(self.affine_c.weight, mode='fan_in')
        self.affine_a.bias.data.fill_(0)
        self.affine_b.bias.data.fill_(0)
        # self.affine_c.bias.data.fill_(0)

    def forward(self, x):

        # x.shape = [b x c x h x w]
        # img = torch.narrow(x, 1, 0, 3)
        # view_position = torch.flatten(torch.narrow(x, 1, 3, 1), 2, 3) # [b x 1 x h x w] -> [b x 1 x h*w]
        A = self.d121(x) # dim size of img_size // 32 x img_size // 32 x 1024

        a_g = self.avgpool(A)
        a_g = a_g.view(a_g.size(0), -1)
        v_g = F.relu(self.affine_b(self.dropout(a_g)))

        V = A.view(A.size(0), A.size(1), -1).transpose(1,2)
        V = F.relu(self.affine_a(self.dropout(V)))

        # view_position = F.relu(self.affine_c(self.dropout(view_position)))
        # V = torch.cat((V, view_position), 1)

        return V, v_g

class SentenceDecoder(nn.Module):
    def __init__(self, hidden_dim, img_feature_dim=256):
        super(SentenceDecoder, self).__init__()
        self.lstmCell = nn.LSTMCell(img_feature_dim, hidden_dim)
        self.topic = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=img_feature_dim),
            nn.ReLU()
        )
        self.stop = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, v_hat, states):
        h_t, c_t = self.lstmCell(v_hat, states)

        t = self.topic(h_t)
        u = self.stop(h_t)

        return u, t, (h_t, c_t)

class WordDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, img_feature_size, word_vectors, embedd_size=256):
        super(WordDecoder, self).__init__()
        wv = torch.from_numpy(word_vectors).float()
        self.embedding = nn.Embedding.from_pretrained(wv, freeze=not self.training)
        self.adaptive = AdaptiveAttention(embedd_size, hidden_size, img_feature_size, vocab_size)

    def forward(self, V, v_g, topic_vector, report, state):
        embeddings = self.embedding(report)
        x = torch.cat((embeddings, v_g, topic_vector), dim=1)
        scores, state = self.adaptive(x, V, state)
        return scores, state

class ModelFactory:
    def __init__(self, args, word_vectors):
        self.args = args
        self.word_vectors = word_vectors

    def get_models(self):
        img_enc = ImageEncoder(self.args.embedd_size, self.args.hidden_size, \
                                self.args.img_size)
        sentence_dec = SentenceDecoder(self.args.hidden_size)
        word_dec = WordDecoder(self.args.vocab_size, self.args.hidden_size, \
                                self.args.img_feature_size, self.word_vectors)
        if self.args.parallel:
            img_enc = nn.DataParallel(img_enc, device_ids=self.args.gpus)
            sentence_dec = nn.DataParallel(sentence_dec, \
                                device_ids=self.args.gpus)
            word_dec = nn.DataParallel(word_dec, device_ids=self.args.gpus)


        img_enc.to(self.args.device)
        sentence_dec.to(self.args.device)
        word_dec.to(self.args.device)
        return img_enc, sentence_dec, word_dec

    def save_models(self, encoder, sentence_decoder, word_decoder, epoch, \
                    optimizer, loss):
        path = "save/"
        torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'sentence_decoder_state_dict': sentence_decoder.state_dict(),
                'word_decoder_state_dict': word_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path + self.args.model_name+".pth"
        )

    def get_params(self, img_enc, sen_dec, word_dec):
        if self.args.parallel:
            params = list(img_enc.module.affine_a.parameters()) \
                    + list(img_enc.module.affine_b.parameters())
        else:
            params = list(img_enc.affine_a.parameters()) \
                    + list(img_enc.affine_b.parameters())
        params = params + list (sen_dec.parameters()) \
                + list(word_dec.parameters())
        return params

    def load_models(self, img_enc, sen_dec, word_dec, optimizer=None):
        model_dict = torch.load(\
                        'save/{model}.pth'.format(model=self.args.model_name), \
                        map_location=self.args.device
                    )
        img_enc.load_state_dict(model_dict['encoder_state_dict'])
        sen_dec.load_state_dict(model_dict['sentence_decoder_state_dict'])
        word_dec.load_state_dict(model_dict['word_decoder_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(model_dict['optimizer_state_dict'])
