import torch
import torch.nn as nn
from torch.nn import functional as F

from tensorboardX import SummaryWriter

from Module.Normalization import ConditionalNorm, SpectralNorm

class GResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=None,
                 padding=1, stride=1, n_class=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False):
        super().__init__()

        if kernel_size is None:
            kernel_size = [3, 3]

        # TODO gain = 2 ** 0.5

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

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.CBNorm1 = ConditionalNorm(in_channel, 148) # const number of class label!
            self.CBNorm2 = ConditionalNorm(out_channel, 148)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            # print('condition',condition.size()) #condition torch.Size([4, 148])
            out = self.CBNorm1(out, condition)

        out = self.activation(out)

        if self.upsample:
            # TODO different form papers
            out = F.interpolate(out, scale_factor=2)

        out = self.conv0(out)

        if self.bn:
            out = self.CBNorm2(out, condition)

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

    gResBlock = GResBlock(20, 100, [3, 3])
    x = torch.rand([4, 20, 64, 64])
    condition = torch.rand([4, 148]) # const 148
    y = gResBlock(x, condition)
    print(gResBlock)
    print(x.size())
    print(y.size())

    with SummaryWriter(comment='gResBlock') as w:
        w.add_graph(gResBlock, [x, condition, ])