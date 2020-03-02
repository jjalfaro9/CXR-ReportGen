import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""
Optional: Your code here
"""

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        densenet121 = models.densenet121(pretrained=True, progress=True)
        # d121 drops the last layer of densenet121,
        #   which contains global average pooling followed by softmax classifier
        self.d121 = nn.Sequential(
            *list(densenet121.features.children())[:-1]
        )

    def forward(self, x):
        features = self.d121(x)
        # need to project to an word embedding of dimensionality d
            # Appendix A: d = 256 with dropout of p = 0.5
        # compute mean visual features by averaging all local visual features

        # view position embeddings (one-hot vector) are concatenated with image embedding to form input to later decoders
        # TODO: does this concatenation happen here or has already been reflected in the input (x) to forward
        return features

class SentenceDecoder(nn.Module):
    def __init__(self, topic_dim, hidden_dim, img_feature_dim):
        super(SentenceDecoder, self).__init__()
        self.topic_dim = topic_dim
        self.hidden_dim = hidden_dim
        self.img_feature_dim = img_feature_dim
        self.lstm = nn.LSTM(input_size=img_feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.topic = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=topic_dim),
            nn.ReLU()
        )
        self.stop = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, img_feature_vector, states):
        img_feature_vector = img_feature_vector.unsqueeze(1)
        h, c = self.lstm(img_feature_vector, states)

        t = self.topic(h)
        u = self.stop(h)
        return u, t, c


class WordDecoder(nn.Module):
    def __init__(self):
        super(WordDecoder, self).__init__()
        self.embedding = nn.Embedding()

    def forward(self):

# the following classes are used to implement adaptive attention :P

class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, prev_h, prev_c):

        h, c = self.lstm_cell(x, (prev_h, prev_c))
        g_t = F.sigmoid(self.x_gate(x) + self.h_gate(prev_h))
        s_t = g_t * F.tanh(c)
        return h, c, s_t

class AdaptiveAttention(nn.Module):
    def __init__(self, ):
        self.affine_v = nn.Linear()
        self.affine_g = nn.Linear()
        self.affine_h = nn.Linear()
        self.affine_s = nn.Linear()

    def forward(self, V, h_t, s_t):
        content_v = self.affine_v(V).unsqueeze(1) + self.affine_g(h_t).unsqueeze(2)
        z_t = self.affine_h(F.tanh(content_v)).squeeze(3)
        alpha_t = F.softmax(z_t)

        content_s = self.affine_s(s_t) + self.aggine_g(h_t)
        z_t_extended = self.affine_h(F.tanh(content_s))

        extended = torch.cat((z_t, z_t_extended), dim=2)
        alpha_hat_t = F.softmax(extended)
        beta_t = alpha_hat_t[-1]

        c_t = torch.bmm(alpha_t, V).squeeze(2)
        c_hat_t = beta_t * s_t + (1-beta_t) * c_t

        return c_hat_t, alpha_t, beta_t

if __name__ == '__main__':
    from torchsummary import summary
    imageEncoder = ImageEncoder()
    print(summary(imageEncoder, (3, 256, 256)))
