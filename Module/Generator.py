import torch
import torch.nn as nn
from torch.nn import functional as F

from tensorboardX import SummaryWriter

from Module.GResBlock import GResBlock
from Module.Normalization import SpectralNorm
from Module.ConvGRU import ConvGRU
from Module.Attention import SelfAttention, SeparableAttn
# from Module.CrossReplicaBN import ScaledCrossReplicaBatchNorm2d

class Generator(nn.Module):

    def __init__(self, in_dim=120, latent_dim=4, n_class=4, ch=32, n_frames=48, hierar_flag=False):
        super().__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.n_class = n_class
        self.ch = ch
        self.hierar_flag = hierar_flag
        self.n_frames = n_frames

        self.embedding = nn.Embedding(n_class, in_dim)

        self.affine_transfrom = nn.Linear(in_dim * 2, latent_dim * latent_dim * 8 * ch)

        # self.self_attn = SelfAttention(8 * ch)

        # self.conv = nn.ModuleList([GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2),
        #                            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2),
        #                            GResBlock(8 * ch, 4 * ch, n_class=in_dim * 2),
        #                            SeparableAttn(4 * ch),
        #                            GResBlock(4 * ch, 2 * ch, n_class=in_dim * 2)])
        # self.convGRU = ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3)

        self.conv = nn.ModuleList([
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2, upsample_factor=1),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2, upsample_factor=1),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2),
            ConvGRU(8 * ch, hidden_sizes=[8 * ch, 16 * ch, 8 * ch], kernel_sizes=[3, 5, 3], n_layers=3),
            # ConvGRU(8 * ch, hidden_sizes=[8 * ch, 8 * ch], kernel_sizes=[3, 3], n_layers=2),
            GResBlock(8 * ch, 8 * ch, n_class=in_dim * 2, upsample_factor=1),
            GResBlock(8 * ch, 4 * ch, n_class=in_dim * 2),
            ConvGRU(4 * ch, hidden_sizes=[4 * ch, 8 * ch, 4 * ch], kernel_sizes=[3, 5, 5], n_layers=3),
            # ConvGRU(4 * ch, hidden_sizes=[4 * ch, 4 * ch], kernel_sizes=[3, 5], n_layers=2),
            GResBlock(4 * ch, 4 * ch, n_class=in_dim * 2, upsample_factor=1),
            GResBlock(4 * ch, 2 * ch, n_class=in_dim * 2)
        ])

        # TODO impl ScaledCrossReplicaBatchNorm
        # self.ScaledCrossReplicaBN = ScaledCrossReplicaBatchNorm2d(1 * chn)

        self.colorize = SpectralNorm(nn.Conv2d(2 * ch, 3, kernel_size=(3, 3), padding=1))


    def forward(self, x, class_id):

        if self.hierar_flag is True:
            noise_emb = torch.split(x, self.in_dim, dim=1)
        else:
            noise_emb = x

        class_emb = self.embedding(class_id)

        if self.hierar_flag is True:
            y = self.affine_transfrom(torch.cat((noise_emb[0], class_emb), dim=1)) # B x (2 x ld x ch)
        else:
            y = self.affine_transfrom(torch.cat((noise_emb, class_emb), dim=1)) # B x (2 x ld x ch)

        y = y.view(-1, 8 * self.ch, self.latent_dim, self.latent_dim) # B x ch x ld x ld

        for k, conv in enumerate(self.conv):
            if isinstance(conv, ConvGRU):

                if k > 0:
                    _, C, W, H = y.size()
                    y = y.view(-1, self.n_frames, C, W, H).contiguous()

                frame_list = []
                for i in range(self.n_frames):
                    if k == 0:
                        if i == 0:
                            frame_list.append(conv(y))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y, frame_list[i - 1]))
                    else:
                        if i == 0:
                            frame_list.append(conv(y[:,0,:,:,:].squeeze(1)))  # T x [B x ch x ld x ld]
                        else:
                            frame_list.append(conv(y[:,i,:,:,:].squeeze(1), frame_list[i - 1]))
                frame_hidden_list = []
                for i in frame_list:
                    frame_hidden_list.append(i[-1].unsqueeze(0))
                y = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld

                y = y.permute(1, 0, 2, 3, 4).contiguous() # B x T x ch x ld x ld
                # print(y.size())
                B, T, C, W, H = y.size()
                y = y.view(-1, C, W, H)

            elif isinstance(conv, GResBlock):
                condition = torch.cat([noise_emb, class_emb], dim=1)
                condition = condition.repeat(self.n_frames,1)
                y = conv(y, condition) # BT, C, W, H

        y = F.relu(y)
        y = self.colorize(y)
        y = torch.tanh(y)

        BT, C, W, H = y.size()
        y = y.view(-1, self.n_frames, C, W, H) # B, T, C, W, H

        return y

        # if torch.cuda.is_available():
        #     frame_list = torch.empty((0,)).cuda()  # initialization similar to frame_list = []
        # else:
        #     frame_list = torch.empty((0,))
        #
        # for i in range(self.n_frames):
        #     if i == 0:
        #         frame_list = self.convGRU(y) # T x [B x ch x ld x ld]
        #     else:
        #         frame_list = torch.stack((frame_list, self.convGRU(y, frame_list[i-1])))
        #
        # frame_hidden_list = torch.empty((0,)).cuda()
        # for i in frame_list.size()[0]:
        #     if i == 0:
        #         frame_hidden_list = 1
        #     frame_hidden_list.append(i[-1].unsqueeze(0))

        # frame_list = []
        # for i in range(self.n_frames):
        #     if i == 0:
        #         frame_list.append(self.convGRU(y))  # T x [B x ch x ld x ld]
        #     else:
        #         frame_list.append(self.convGRU(y, frame_list[i - 1]))
        #
        # frame_hidden_list = []
        # for i in frame_list:
        #     frame_hidden_list.append(i[-1].unsqueeze(0))
        # y = torch.cat(frame_hidden_list, dim=0) # T x B x ch x ld x ld
        # y = y.permute(1, 2, 0, 3, 4) # B x ch x T x ld x ld
        # y = self.self_attn(y)  # B x ch x T x ld x ld
        #
        # y = y.permute(0, 2, 1, 3, 4).contiguous() # B x T x ch x ld x ld
        #
        # # the time axis is folded into the batch axis before the forward pass, which applying ResNet to all frames indivudually
        # y = y.view(-1, 8 * self.ch, self.latent_dim, self.latent_dim) # (B x T) x ch x ld x ld
        #
        # frame = y
        # for j, conv in enumerate(self.conv):
        #     if isinstance(conv, GResBlock):
        #         condition = torch.cat([noise_emb, class_emb], dim=1)
        #         condition = condition.repeat(self.n_frames, 1)
        #         frame = conv(frame, condition)
        #     else:
        #         BT, C, W, H = frame.size()
        #         frame = frame.view(-1, self.n_frames, C, W, H).transpose(2, 1) # B, C, T, W, H
        #         frame = conv(frame)
        #         frame = frame.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, W, H) # BT, C, W, H
        #
        # frame = F.relu(frame)
        # frame = self.colorize(frame)
        # frame = torch.tanh(frame)
        #
        # BT, C, W, H = frame.size()
        # frame = frame.view(-1, self.n_frames, C, W, H) # B, T, C, W, H

        # return frame


if __name__ == "__main__":

    batch_size = 5
    in_dim = 120
    n_class = 4
    n_frames = 4

    x = torch.randn(batch_size, in_dim).cuda()
    class_label = torch.randint(low=0, high=3, size=(batch_size,)).cuda()
    generator = Generator(in_dim, n_class=n_class, ch=3, n_frames=n_frames).cuda()
    y = generator(x, class_label)

    print(x.size())
    print(y.size())
