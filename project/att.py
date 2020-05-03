import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class VisualSentinel(nn.Module):
    def __init__(self, input, hidden):
        super(VisualSentinel, self).__init__()
        self.Wx = nn.Linear(input, hidden, bias=False)
        self.Wh = nn.Linear(hidden, hidden, bias=False)
        self.__initWeightss()

    def __initWeightss(self):
        init.kaiming_uniform_(self.Wx.weight, mode='fan_in')
        init.kaiming_uniform_(self.Wh.weight, mode='fan_in')

    def forward(self, x, prev_hidden, mem):
        g = torch.sigmoid(self.Wx(x) + self.Wh(prev_hidden))
        s = g * torch.tanh(mem)
        return s

class Attention(nn.Module):
    def __init__(self, input, hidden_size, K):
        super(Attention, self).__init__()
        self.vSentinel = VisualSentinel(input, hidden_size)
        self.Wv = nn.Linear(hidden_size, K, bias=False)
        self.Wg = nn.Linear(hidden_size, K, bias=False)
        self.Wh = nn.Linear(K, 1, bias=False)
        self.Ws = nn.Linear(hidden_size, K, bias=False)
        self.__initWeightss()

    def __initWeightss(self):
        init.kaiming_uniform_(self.Wv.weight, mode='fan_in')
        init.kaiming_uniform_(self.Wg.weight, mode='fan_in')
        init.kaiming_uniform_(self.Wh.weight, mode='fan_in')
        init.kaiming_uniform_(self.Ws.weight, mode='fan_in')

    def forward(self, V, cur_hidden, cur_mem, prev_hidden, x):
        content_v = self.Wv(V) + self.Wg(cur_hidden).unsqueeze(1)
        z = self.Wh(torch.tanh(content_v)).squeeze(2)
        alpha = F.softmax(z, dim=1).unsqueeze(1)
        c_t = torch.bmm(alpha, V).squeeze(1)

        s = self.vSentinel(x, prev_hidden, cur_mem)
        mln = self.Ws(s) + self.Wg(cur_hidden)
        how_much = self.Wh(torch.tanh(mln))
        alpha_hat = F.softmax(torch.cat((z, how_much), dim=1), dim=1)
        beta = alpha_hat[:, -1]

        c_t_hat = beta[:, None] * s + (1-beta)[:, None] * c_t
        return c_t_hat

class AdaptiveAttention(nn.Module):
    def __init__(self, input, hidden, K, vocab_size):
        # 256, 512, 64, vocab
        super(AdaptiveAttention, self).__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        # The 3 is because we have [word_emb ; global_avg_feats; topic_vec ]
        self.atten = Attention(3 * input, hidden, K)
        self.lstmCell = nn.LSTMCell(3 * input, hidden)
        self.mlp = nn.Linear(hidden, vocab_size)
        self.__initWeightss()

    def __initWeightss(self):
        init.kaiming_uniform_(self.mlp.weight, mode='fan_in')

    def forward(self, x, V, state):
        # x = [wt; vg; topic_vec] => batch x 256 * 3
        # V = batch x 64 x 512
        h_t_prev = state[0]
        h_t, mem_t = self.lstmCell(x, state)
        c_t = self.atten(V, h_t, mem_t, h_t_prev, x)
        scores = self.mlp(c_t + h_t)
        return scores, (h_t, mem_t)
