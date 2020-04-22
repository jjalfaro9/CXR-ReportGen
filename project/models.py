import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import init

from adaptiveattention import AdaptiveBlock

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

        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.affine_a.weight, mode='fan_in')
        init.kaiming_uniform_(self.affine_b.weight, mode='fan_in')
        self.affine_a.bias.data.fill_(0)
        self.affine_b.bias.data.fill_(0)

    def forward(self, x):
        print("\tIn Model: input size", x.size())
        # TODO: once we don't convert to RGB, figure out channel position and extract it
        A = self.d121(x) # dim size of img_size // 32 x img_size // 32 x 1024

        a_g = self.avgpool(A)
        a_g = a_g.view(a_g.size(0), -1)
        V = A.view(A.size(0), A.size(1), -1).transpose(1,2)
        V = F.relu(self.affine_a(self.dropout(V)))

        v_g = F.relu(self.affine_b(self.dropout(a_g)))

        # print("\tIn Model: output size", V.size(), v_g.size())
        return V, v_g

class SentenceDecoder(nn.Module):
    def __init__(self, vocab_dim, hidden_dim, img_feature_dim=256):
        super(SentenceDecoder, self).__init__()
        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.img_feature_dim = img_feature_dim
        self.lstm = nn.LSTM(input_size=img_feature_dim*2, hidden_size=hidden_dim, batch_first=True)
        self.topic = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=img_feature_dim),
            nn.ReLU()
        )
        self.stop = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, img_feature_vector, states):
        # TODO only do this when running on multiple GPUs
        if states is not None:
            s0 = states[0].permute(1,0,2).contiguous()
            s1 = states[1].permute(1,0,2).contiguous()
            states = (s0, s1)
        output, (h, c) = self.lstm(img_feature_vector, states)

        t = self.topic(output)
        u = self.stop(output)
        return u, t, (h.permute(1,0,2).contiguous(), c.permute(1,0,2).contiguous())

class WordDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, img_feature_size, embedd_size=256):
        super(WordDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedd_size)
        self.lstm = nn.LSTM(input_size=embedd_size, hidden_size=hidden_size, batch_first=True)
        self.adaptive = AdaptiveBlock(embedd_size, hidden_size, vocab_size, img_feature_size)
        #if torch.cuda.device_count() > 1:
            #self.adaptive = nn.DataParallel(self.adaptive)
        self.hidden_size = hidden_size

    def forward(self, V, v_g, topic_vector, report):
        # print("\tIn Model: input size", report.size())
        embeddings = self.embedding(report)
        x = torch.cat((embeddings, v_g.unsqueeze(1), topic_vector), dim=1)

        h_t, cells = self.lstm(x)
        scores, atten_weights, beta = self.adaptive(x, h_t, cells, V)

        return scores[:, -1], atten_weights, beta

class Encoder2Decoder(nn.Module):
    def __init__( self, embedd_size, vocab_size, hidden_size ):
        super( Encoder2Decoder, self ).__init__()

        self.hidden_size = hidden_size

        self.image_enc = ImageEncoder(embedd_size, hidden_size)
        self.sentence_dec = SentenceDecoder(vocab_size, hidden_size)
        self.word_dec = WordDecoder(embedd_size, vocab_size, hidden_size)

    def forward(self, images, captions, lengths):
        V, v_g = self.image_enc( images )

        u = 1
        states = None
        # TODO: WIll list affect placement on GPU?
        sentence_scores = []

        # loop through sentence decoder until u > 0.5
        while (u > 0.5):
            u, t, states = self.sentence_dec(v_g, states)
            scores, states, atten_weights, beta = self.word_dec(V, v_g, t)
            packed_scores = pack_padded_sequence(scores, lengths, batch_first=True )
            sentence_scores.append(packed_scores)

        return sentence_scores

    def sample(images, max_len=100):

        V, v_g = self.image_enc( images )

        u = 1
        states = None
        # TODO: WIll list affect placement on GPU?
        sentences = []

        # loop through sentence decoder until u > 0.5
        while (u > 0.5):
            u, t, states = self.sentence_dec(v_g, states)
            scores, states, atten_weights, beta = self.word_dec(V, v_g, t)
            predicted = scores.max( 2 )[ 1 ] # argmax
            sentences.append(predicted)

        sentences = torch.cat( sentences, dim=1 )
        return sentences

if __name__ == '__main__':
    from torchsummary import summary
    imageEncoder = ImageEncoder(256, 64)
    print(summary(imageEncoder, (3, 128, 128)))
