import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from tensorboardX import SummaryWriter

from Module.Normalization import SpectralNorm

class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def init_conv(conv, glu=True):
        init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()

    def __init__(self, in_dim, activation=F.relu, pooling_factor = 2):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = SpectralNorm( # SpectralNorm is added
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=in_dim // 8,
                kernel_size=1
            )
        )

        self.key_conv = SpectralNorm(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=in_dim // 8,
                kernel_size=1
            )
        )
        self.value_conv = SpectralNorm(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=in_dim,
                kernel_size=1
            )
        )

        self.pooling = nn.MaxPool2d(pooling_factor, pooling_factor)
        self.pooling_factor = pooling_factor * 2

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):

        m_batchsize, C, width, height = x.size()
        query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x N x C

        key = self.key_conv(x)  # B x C x H x W
        key = self.pooling(key).view(m_batchsize, -1, width * height // self.pooling_factor) # B x C x (N // 4)

        dist = torch.bmm(query, key) # B x N x (N // 4)
        attn_score = self.softmax(dist) # B x N x (N // 4)

        value = self.value_conv(x)
        value = self.pooling(value).view(m_batchsize, -1, width * height // self.pooling_factor)  # B x C x (N // 4)

        out = torch.bmm(value, attn_score.permute(0, 2, 1)) # B x C x N
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


if __name__ == "__main__":

    self_attn = SelfAttention(64)
    print(self_attn)

    x = torch.rand(1, 64, 128, 128)
    y = self_attn(x)

    with SummaryWriter(comment='self-attention') as w:
        w.add_graph(self_attn, [x,])