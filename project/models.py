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

    def forward(self, V, v_g, topic_vector, report):
        embeddings = self.embedding(report)
        x = torch.cat((embeddings, v_g.unsqueeze(1), topic_vector.unsqueeze(1)), dim=1)
        scores = self.adaptive(x, V)

        return scores.permute(0, 2, 1)[:, :, :report.size(1)]

if __name__ == '__main__':
    from torchsummary import summary
    imageEncoder = ImageEncoder(256, 64)
    print(summary(imageEncoder, (3, 128, 128)))
