import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

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
        g_t = torch.sigmoid(self.x_gate(x) + self.h_gate(prev_h))
        s_t = g_t * torch.tanh(c)
        return h, c, s_t

class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, img_feature_size):
        super(AdaptiveAttention, self).__init__()
        # self.affine_v = nn.Linear(hidden_size, 8*8, bias=False)
        # self.affine_g = nn.Linear(hidden_size, 8*8, bias=False)
        # self.affine_h = nn.Linear(8*8, 1, bias=False)
        # self.affine_s = nn.Linear(hidden_size, 8*8, bias=False)

        # self.init_weights()
        self.sen_affine = nn.Linear(hidden_size, hidden_size)
        self.sen_att = nn.Linear(hidden_size, img_feature_size)
        self.h_affine = nn.Linear(hidden_size, hidden_size)
        self.h_att = nn.Linear(hidden_size, img_feature_size)
        self.v_att = nn.Linear(hidden_size, img_feature_size)
        self.alphas = nn.Linear(img_feature_size, 1)
        self.context_hidden = nn.Linear(hidden_size, hidden_size)

    def init_weights(self):
        init.xavier_uniform_(self.affine_v.weight)
        init.xavier_uniform_(self.affine_g.weight)
        init.xavier_uniform_(self.affine_h.weight)
        init.xavier_uniform_(self.affine_s.weight)

    def forward(self, V, h_t, s_t):
    # def forward(self, spatial_image, decoder_out, st):
        # content_v = self.affine_v(V).unsqueeze(1) + self.affine_g(h_t).unsqueeze(2)
        # z_t = self.affine_h(torch.tanh(content_v)).squeeze(3)
        # alpha_t = F.softmax(z_t)

        # content_s = self.affine_s(s_t) + self.affine_g(h_t)
        # z_t_extended = self.affine_h(torch.tanh(content_s)).unsqueeze(2)

        # extended = torch.cat((z_t, z_t_extended), dim=2)
        # alpha_hat_t = F.softmax(extended)
        # beta_t = alpha_hat_t[-1]

        # c_t = torch.bmm(alpha_t, V).squeeze(2)
        # c_hat_t = beta_t * s_t + (1-beta_t) * c_t

        # return c_hat_t, alpha_t, beta_t

        num_pixels = V.shape[1]
        visual_attn = self.v_att(V)           # (batch_size,num_pixels,att_dim)
        sentinel_affine = F.relu(self.sen_affine(s_t))     # (batch_size,hidden_size)
        sentinel_attn = self.sen_att(sentinel_affine)     # (batch_size,att_dim)

        hidden_affine = torch.tanh(self.h_affine(h_t))    # (batch_size,hidden_size)
        hidden_attn = self.h_att(hidden_affine)               # (batch_size,att_dim)

        hidden_resized = hidden_attn.unsqueeze(1).expand(hidden_attn.size(0), num_pixels + 1, hidden_attn.size(1))

        concat_features = torch.cat([V, sentinel_affine.unsqueeze(1)], dim = 1)   # (batch_size, num_pixels+1, hidden_size)
        attended_features = torch.cat([visual_attn, sentinel_attn.unsqueeze(1)], dim = 1)     # (batch_size, num_pixels+1, att_dim)

        attention = torch.tanh(attended_features + hidden_resized)    # (batch_size, num_pixels+1, att_dim)

        alpha = self.alphas(attention).squeeze(2)                   # (batch_size, num_pixels+1)
        alpha_t = torch.softmax(alpha, dim=1)                              # (batch_size, num_pixels+1)

        context = (concat_features * alpha_t.unsqueeze(2)).sum(dim=1)       # (batch_size, hidden_size)
        beta_t = alpha_t[:,-1].unsqueeze(1)                       # (batch_size, 1)

        c_hat_t = torch.tanh(self.context_hidden(context + hidden_affine))

        return c_hat_t, alpha_t, beta_t

class AdaptiveBlock(nn.Module):
    def __init__(self, embedd_size, hidden_size, vocab_size, img_feature_size):
        super(AdaptiveBlock, self).__init__()
        self.sentinel = AdaptiveLSTMCell(embedd_size, hidden_size)
        self.attention = AdaptiveAttention(hidden_size, img_feature_size)

        self.mlp = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(0.5)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedd_size = embedd_size
        self.img_feature_size = img_feature_size

        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def init_hidden_state(self, x):
        h = torch.zeros(x.shape[0], 512).to(x.device)
        c = torch.zeros(x.shape[0], 512).to(x.device)
        return h, c

    def forward(self, x, hiddens, cells, V):
        h, c = self.init_hidden_state(x)

        # TO DO: TIMESTEPS
        batch_size = x.shape[0]
        # TODO: Lengths?
        decode_length = x.shape[1]
        scores = torch.zeros(batch_size, decode_length, self.vocab_size).to(x.device)
        atten_weights = torch.zeros(batch_size, decode_length, self.img_feature_size + 1).to(x.device)
        betas = torch.zeros(batch_size, decode_length, 1).to(x.device)

        for timestep in range(decode_length):
            current_input = x[:, timestep, :]
            h, c, sentinel = self.sentinel(current_input, h, c)
            c_hat, atten_weight, beta = self.attention(V, h, sentinel)
            score = self.mlp(self.dropout(c_hat + h))

            scores[:, timestep, :] = score
            atten_weights[:, timestep, :] = atten_weight
            betas[:, timestep, :] = beta

        return scores, atten_weights, betas
