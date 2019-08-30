import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from tensorboardX import SummaryWriter

class SeparableAttn(nn.Module):

    def __init__(self, in_dim, activation=F.relu, pooling_factor=1, padding_mode='constant', padding_value=0):
        super().__init__()
        self.model = nn.Sequential(
            SeparableAttnCell(in_dim, 'T', activation, pooling_factor, padding_mode, padding_value),
            SeparableAttnCell(in_dim, 'W', activation, pooling_factor, padding_mode, padding_value),
            SeparableAttnCell(in_dim, 'H', activation, pooling_factor, padding_mode, padding_value)
        )

    def forward(self, x):

        return self.model(x)


class SeparableAttnCell(nn.Module):

    def __init__(self, in_dim, attn_id = None, activation=F.relu, pooling_factor=1, padding_mode='constant', padding_value=0):
        super().__init__()
        self.attn_id = attn_id
        self.activation = activation

        self.query_conv = nn.Conv3d(
                                in_channels=in_dim,
                                out_channels=in_dim // 8,
                                kernel_size=1
                            )

        self.key_conv = nn.Conv3d(
                            in_channels=in_dim,
                            out_channels=in_dim // 8,
                            kernel_size=1
                        )
        self.value_conv = nn.Conv3d(
                            in_channels=in_dim,
                            out_channels=in_dim,
                            kernel_size=1
                        )

        self.pooling = nn.MaxPool3d(kernel_size=pooling_factor)
        self.pooling_factor = pooling_factor ** 3

        self.padding_mode = padding_mode
        self.padding_value = padding_value

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def init_conv(self, conv, glu=True):
        init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()

    def forward(self, x):

        m_batchsize, C, time, width, height = x.size()

        # TODO attention space consumption
        # query = self.query_conv(x).view(m_batchsize, -1, time * width).permute(0, 2, 1)  # B x (TW) x (CH)
        #
        # key = self.key_conv(x)  # B x C x T x H x W
        # key = self.pooling(key).view(m_batchsize, -1, time * height // self.pooling_factor)  # B x (CW) x (TH // 8)
        #
        # if height < width:
        #     query = F.pad(query, [0, C * (width - height)], self.padding_mode, self.padding_value)
        # else:
        #     key = F.pad(key, [0, 0, 0, C * (height - width)], self.padding_mode, self.padding_value)

        if self.attn_id == 'T':
            attn_dim = time
        elif self.attn_id == 'H':
            attn_dim = height
        else:
            attn_dim = width

        query = self.query_conv(x).view(m_batchsize, attn_dim, -1) # B x T x (WHC)
        key = self.key_conv(x)  # B x C x T x H x W
        key = self.pooling(key).view(m_batchsize, -1, attn_dim // self.pooling_factor)  # B x (WHC) x (T // 4)


        dist = torch.bmm(query, key)  # B x T x (T // 4)
        attn_score = self.softmax(dist)  # B x T x (T // 4)

        value = self.value_conv(x)
        value = self.pooling(value).view(m_batchsize, -1, attn_dim // self.pooling_factor)  # B x (WHC) x (T // 4)

        out = torch.bmm(value, attn_score.permute(0, 2, 1))  # B x (WHC) x T
        out = out.view(m_batchsize, C, width, height, time)
        out = out.permute(0, 1, 4, 2, 3)

        out = self.gamma * out + x
        return out


class SelfAttention(nn.Module):

    def __init__(self, in_dim, activation=F.relu, pooling_factor = 2): # TODO for better compability

        super(SelfAttention, self).__init__()
        self.activation = activation

        self.query_conv = nn.Conv3d(
                            in_channels=in_dim,
                            out_channels=in_dim // 8,
                            kernel_size=1
                        )

        self.key_conv = nn.Conv3d(
                            in_channels=in_dim,
                            out_channels=in_dim // 8,
                            kernel_size=1
                        )

        self.value_conv = nn.Conv3d(
                            in_channels=in_dim,
                            out_channels=in_dim,
                            kernel_size=1
                        )

        self.pooling = nn.MaxPool3d(kernel_size=pooling_factor)
        self.pooling_factor = pooling_factor ** 3

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)


    def init_conv(self, conv, glu=True):
        init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()


    def forward(self, x):

        if len(x.size()) == 4:
            m_batchsize, C, width, height = x.size()
            time = 1
        else:
            m_batchsize, C, time, width, height = x.size()

        N = time * width * height


        query = self.query_conv(x).view(m_batchsize, -1, N).permute(0, 2, 1)  # B x N x C

        key = self.key_conv(x)  # B x C x H x W
        key = self.pooling(key).view(m_batchsize, -1, N // self.pooling_factor) # B x C x (N // 4)

        dist = torch.bmm(query, key) # B x N x (N // 4)
        attn_score = self.softmax(dist) # B x N x (N // 4)

        value = self.value_conv(x)
        value = self.pooling(value).view(m_batchsize, -1, N // self.pooling_factor)  # B x C x (N // 4)

        out = torch.bmm(value, attn_score.permute(0, 2, 1)) # B x C x N

        if len(x.size()) == 4:
            out = out.view(m_batchsize, C, width, height)
        else:
            out = out.view(m_batchsize, C, time, width, height)

        out = self.gamma * out + x
        return out


if __name__ == "__main__":

    self_attn = SelfAttention(16) # no less than 8
    print(self_attn)

    n_frames = 4

    x = torch.rand(1, 16, n_frames, 32, 32)
    y = self_attn(x)
    print(x.size())
    print(y.size())

    # with SummaryWriter(comment='self-attention') as w:
    #     w.add_graph(self_attn, [x,])

    del x, y


    sepa_attn = SeparableAttn(64)
    print(sepa_attn)
    x = torch.rand(1, 64, 3, 128, 128)
    y = sepa_attn(x)
    print(x.size())
    print(y.size())

    # with SummaryWriter(comment='separable-attention') as w:
    #     w.add_graph(self_attn, [x,])