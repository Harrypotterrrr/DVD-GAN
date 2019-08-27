import torch
import torch.nn as nn
from torch.nn import functional as F

from tensorboardX import SummaryWriter

from Module.Normalization import ConditionalNorm, SpectralNorm

class GResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=None,
                 padding=1, stride=1, n_class=96, n_frames=48, bn=True,
                 activation=F.relu, upsample=True, downsample=False):
        super().__init__()

        self.in_channel = in_channel
        self.n_frames = n_frames
        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn

        if kernel_size is None:
            kernel_size = [3, 3]

        self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))
        self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                                   1, 1, 0))
            self.skip_proj = True

        if bn:
            self.CBNorm1 = ConditionalNorm(in_channel, n_class) # 2 x noise.size[1]
            self.CBNorm2 = ConditionalNorm(out_channel, n_class)

    def forward(self, input, condition=None):

        batch_size, T, C, W, H = input.size()
        print(batch_size, T,C,W,H)

        out = input

        if self.bn:
            out = out.view(-1, C, W, H)
            out = self.CBNorm1(out, condition)
            out = out.view(-1, T, C, W, H)

        out = self.activation(out)

        if self.upsample:
            # TODO different form papers
            out = F.interpolate(out, scale_factor=2)

        out = self.conv0(out)

        if self.bn:
            out = out.view(batch_size, C, W * 2, H * 2)
            out = self.CBNorm2(out, condition)
            out = out.view(batch_size*T, C, W * 2, H * 2)

        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.interpolate(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip


if __name__ == "__main__":

    n_class = 96

    gResBlock = GResBlock(3, 100, [3, 3], n_frames=20)
    x = torch.rand([4, 20, 3, 64, 64])
    condition = torch.rand([4, n_class])
    y = gResBlock(x, condition)
    print(gResBlock)
    print(x.size())
    print(y.size())

    with SummaryWriter(comment='gResBlock') as w:
        w.add_graph(gResBlock, [x, condition, ])