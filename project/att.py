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
        self.atten = Attention(input, hidden, K)
        self.lstmCell = nn.LSTMCell(input, hidden)
        self.mlp = nn.Linear(hidden, vocab_size)
        self.__initWeightss()

    def __initWeightss(self):
        init.kaiming_uniform_(self.mlp.weight, mode='fan_in')

    def forward(self, x, V):
        # x = [wt; vg; topic_vec] => batch x seq_len x 256
        # V = batch x 64 x 512
        scores = torch.zeros(x.size(0), x.size(1), self.vocab_size).to(x.device)
        h_t = torch.zeros(x.size(0), self.hidden).to(x.device)
        mem_t = torch.zeros(x.size(0), self.hidden).to(x.device)
        for time_step in range(x.size(1)):
            x_t = x[:, time_step, :]
            h_t_prev = h_t
            h_t, mem_t = self.lstmCell(x_t, (h_t, mem_t))
            c_t = self.atten(V, h_t, mem_t, h_t_prev, x_t)
            y_t = self.mlp(c_t + h_t)
            scores[:, time_step, :] = y_t
        return scores
