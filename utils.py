import os
import torch
from torch.nn import init

def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))

def set_device(config):
    if config.gpus == "": # cpu
        return 'cpu', False, ""
    elif torch.cuda.is_available() is False: # cpu
        return 'cpu', False, ""
    elif config.parallel is True and len(config.gpus.split(',')) > 1: # multi gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
        return 'cuda', True, config.gpus
    else: # single gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
        return 'cuda:'+config.gpus, False, config.gpus

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
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def sample_k_frames(data, length, n_sample):

    idx = torch.randint(0, length, (n_sample,))
    srt, idx = idx.sort()
    return data[:, srt, :, :, :]