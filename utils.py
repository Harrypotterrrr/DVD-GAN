import os
import torch
from torch.nn import init
import torch.nn.functional as F

def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))

def set_device(config):

    if config.gpus == "": # cpu
        return 'cpu', False, ""
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(config.gpus)

        if torch.cuda.is_available() is False: # cpu
            return 'cpu', False, ""
        else:
            # gpus = config.gpus.split(',') # if config.gpus is a list
            # gpus = (',').join(list(map(str, range(0, len(gpus))))) # generate a list of string number from 0 to len(config.gpus)
            gpus = list(range(len(config.gpus)))
            if config.parallel is True and len(gpus) > 1: # multi gpus
                return 'cuda:0', True, gpus
            else: # single gpu
                return 'cuda:'+ str(gpus[0]), False, gpus

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()

    x.requires_grad = True
    return x

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def sample_k_frames(data, video_length, k_sample):
    frame_idx = torch.randperm(video_length)
    srt, idx = frame_idx[:k_sample].sort()
    return data[:, srt, :, :, :]

def write_log(writer, log_str, step, ds_loss_real, ds_loss_fake, ds_loss, dt_loss_real, dt_loss_fake, dt_loss, g_loss):

    writer.add_scalar('data/ds_loss_real', ds_loss_real.item(), step)
    writer.add_scalar('data/ds_loss_fake', ds_loss_fake.item(), step)
    writer.add_scalar('data/ds_loss', ds_loss.item(), step)
    writer.add_scalar('data/dt_loss_real', dt_loss_real.item(), step)
    writer.add_scalar('data/dt_loss_fake', dt_loss_fake.item(), step)
    writer.add_scalar('data/dt_loss', dt_loss.item(), step)
    writer.add_scalar('data/g_loss_fake', g_loss.item(), step)

    writer.add_text('logs', log_str, step)

def vid_downsample(data):
    out = data
    B, T, C, H, W = out.size()
    x = F.avg_pool2d(out.view(B * T, C, H, W), kernel_size=2)
    _, _, H, W = x.size()
    x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
    return x
