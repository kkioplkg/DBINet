import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)

        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        self.bn = nn.BatchNorm1d(h_dim * self.k)
        self.fc_adjust = nn.Linear(h_dim * self.k, 1024)
        nn.init.kaiming_normal_(self.fc_adjust.weight)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bi,bj->bij', (v, q))

        if 1 < self.k:
            fusion_logits = self.p_net(fusion_logits.permute(0, 2, 1)).permute(0, 2, 1).mean(dim=1) * self.k
        else:
            fusion_logits = fusion_logits.mean(dim=1)
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)

        v_ = self.v_net(v)
        q_ = self.q_net(q)


        att_maps = torch.einsum('bi,bj->bij', (v_, q_))

        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, v_num * q_num), 1)
            att_maps = p.view(-1, v_num, q_num)

        logits = self.attention_pooling(v_, q_, att_maps)
        logits = self.bn(logits)
        logits = self.fc_adjust(logits)

        return logits, att_maps


class FCNet(nn.Module):

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
