import torch
import torch.nn as nn
from torch.nn import init

class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
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
        super(AdaptiveAttention, self).__init__()
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

        self.mlp = nn.Linear(hidden_size, vocab_size)

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
