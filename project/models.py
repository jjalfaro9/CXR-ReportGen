import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import init

"""
Optional: Your code here
"""

class ImageEncoder(nn.Module):
    def __init__(self, embedd_size, hidden_size):
        super(ImageEncoder, self).__init__()
        densenet121 = models.densenet121(pretrained=True) #, progress=True)
        # d121 drops the last layer of densenet121,
        #   which contains global average pooling followed by softmax classifier
        self.d121 = nn.Sequential(
            *list(densenet121.features.children())[:-1]
        )

        self.avgpool = nn.AvgPool2d(8)
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
        A = self.d121(x) # dim size of 8 x 8 x 1024

        # need to project to an word embedding of dimensionality d
            # Appendix A: d = 256 with dropout of p = 0.5
        # compute mean visual features by averaging all local visual features

        a_g = self.avgpool(A)
        a_g = a_g.view(a_g.size(0), -1)
        V = A.view(A.size(0), A.size(1), -1).transpose(1,2)
        V = F.relu(self.affine_a(self.dropout(V)))

        v_g = F.relu(self.affine_b(self.dropout(a_g)))
        # view position embeddings (one-hot vector) are concatenated with image embedding to form input to later decoders
        # TODO: does this concatenation happen here or has already been reflected in the input (x) to forward

        print("V.shape", V.shape)
        print('v_g.shape', v_g.shape)

        return V, v_g

class SentenceDecoder(nn.Module):
    def __init__(self, vocab_dim, hidden_dim, img_feature_dim=256):
        super(SentenceDecoder, self).__init__()
        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.img_feature_dim = img_feature_dim
        self.lstm = nn.LSTM(input_size=img_feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.topic = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=vocab_dim),
            nn.ReLU()
        )
        self.stop = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, img_feature_vector, states):
        img_feature_vector = img_feature_vector.unsqueeze(1)
        output, (h, c) = self.lstm(img_feature_vector, states)

        t = self.topic(output)
        u = self.stop(output)
        return u, t, (h, c)

class WordDecoder(nn.Module):
    def __init__(self, embedd_size=256, vocab_size, hidden_size):
        super(WordDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedd_size)
        self.lstm = nn.LSTM(embedd_size*2, hidden_size, 1, batch_first=True)
        self.adaptive = AdaptiveBlock(embedd_size, hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, V, v_g, topic_vector):
        embeddings = self.embedding(topic_vector)

        # x_t = [w_t;v_g]
        x = torch.cat((embeddings, v_g.unsqueeze(1).expand_as(embeddings)), dim=2)

        # Hiddens: Batch x seq_len x hidden_size
        # Cells: seq_len x Batch x hidden_size, default setup by Pytorch
        if torch.cuda.is_available():
            hiddens = torch.zeros(x.size(0), x.size(1), self.hidden_size).cuda()
            cells = torch.zeros(x.size(1), x.size(0), self.hidden_size).cuda()
        else:
            hiddens = torch.zeros(x.size(0), x.size(1), self.hidden_size)
            cells = torch.zeros(x.size(1), x.size(0), self.hidden_size)
        
        states = None
        # until we reach max caption length??
        for time_step in range(x.size( 1 )): 
            # Feed in x_t one at a time
            x_t = x[ :, time_step, : ]
            x_t = x_t.unsqueeze( 1 )
            
            h_t, states = self.LSTM( x_t, states )
            
            # Save hidden and cell
            hiddens[ :, time_step, : ] = h_t  # Batch_first
            cells[ time_step, :, : ] = states[ 1 ]    

        # cell: Batch x seq_len x hidden_size
        cells = cells.transpose( 0, 1 )

        # Data parallelism for adaptive attention block 
        # TODO: 4 GPUS
        # if torch.cuda.device_count() > 1:
        #     ids = range( torch.cuda.device_count() )
        #     adaptive_block_parallel = nn.DataParallel( self.adaptive, device_ids=ids )
            
        #     scores, atten_weights, beta = adaptive_block_parallel( x, hiddens, cells, V )
        # else:
        scores, atten_weights, beta = self.adaptive( x, hiddens, cells, V )


        return scores, states, atten_weights, beta

# the following classes are used to implement adaptive attention :P

class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size, bias=False)
        self.h_gate = nn.Linear(hidden_size, hidden_size, bias=False)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.x_gate.weight)
        init.xavier_uniform_(self.h_gate.weight)

    def forward(self, x, prev_h, prev_c):
        h, c = self.lstm_cell(x, (prev_h, prev_c))
        g_t = F.sigmoid(self.x_gate(x) + self.h_gate(prev_h))
        s_t = g_t * F.tanh(c)
        return h, c, s_t

class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size):
        self.affine_v = nn.Linear(hidden_size, 8*8, bias=False)
        self.affine_g = nn.Linear(hidden_size, 8*8, bias=False)
        self.affine_h = nn.Linear(8*8, 1, bias=False)
        self.affine_s = nn.Linear(hidden_size, 8*8, bias=False)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.affine_v.weight)
        init.xavier_uniform_(self.affine_g.weight)
        init.xavier_uniform_(self.affine_h.weight)
        init.xavier_uniform_(self.affine_s.weight)

    def forward(self, V, h_t, s_t):
        content_v = self.affine_v(V).unsqueeze(1) + self.affine_g(h_t).unsqueeze(2)
        z_t = self.affine_h(F.tanh(content_v)).squeeze(3)
        alpha_t = F.softmax(z_t)

        content_s = self.affine_s(s_t) + self.affine_g(h_t)
        z_t_extended = self.affine_h(F.tanh(content_s))

        extended = torch.cat((z_t, z_t_extended), dim=2)
        alpha_hat_t = F.softmax(extended)
        beta_t = alpha_hat_t[-1]

        c_t = torch.bmm(alpha_t, V).squeeze(2)
        c_hat_t = beta_t * s_t + (1-beta_t) * c_t

        return c_hat_t, alpha_t, beta_t

class AdaptiveBlock(nn.Module):
    def __init__(self, embedd_size, hidden_size, vocab_size):
        super(AdaptiveBlock, self).__init__()
        self.sentinel = AdaptiveLSTMCell(embedd_size*2, hidden_size)
        self.attention = AdaptiveAttention(hidden_size)

        self.mlp = nn.Linear(hiden_size, vocab_size)

        self.dropout = nn.Dropout(0.5)
        self.hidden_size = hidden_size

        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def forward(self, x, hiddens, cells, V):
        h0 = self.init_hidden(x.size(0))[0].transpose(0,1)

        if hiddens.size(1) > 1:
            hiddens_t_1 = torch.cat((h0, hiddens[:, :-1, :]), dim=1)
        else:
            hiddens_t_1 = h0

        sentinel = self.sentinel(x, hiddens_t_1, cells)
        c_hat, atten_weights, beta = self.attention(V, hiddens, sentinel)
        scores = self.mlp(self.dropout(c_hat + hiddens))

        return scores, atten_weights, beta

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            return (weight.new(1, bsz, self.hidden_size).zero_().cuda(),
                    weight.new(1, bsz, self.hidden_size).zero_().cuda())
        else:
            return (weight.new(1, bsz, self.hidden_size).zero_(),
                    weight.new(1, bsz, self.hidden_size).zero_())

class Encoder2Decoder(nn.Module):
    def __init__( self, embedd_size, vocab_size, hidden_size ):
        super( Encoder2Decoder, self ).__init__()

        self.hidden_size = hidden_size

        self.image_enc = ImageEncoder(embedd_size, hidden_size)
        self.sentence_dec = SentenceDecoder(vocab_dim, hidden_size)
        self.word_dec = WordDecoder(embedd_size, vocab_size, hidden_size)

    def forward(self, images, captions, lengths):
        # TODO: 4 GPUS
        # if torch.cuda.device_count() > 1:
        #     device_ids = range( torch.cuda.device_count() )
        #     encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
        #     V, v_g = encoder_parallel( images ) 
        # else:
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
    print(summary(imageEncoder, (3, 256, 256)))
