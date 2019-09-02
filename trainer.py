
import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from Module.Generator import Generator
from Module.Discriminators import SpatialDiscriminator, TemporalDiscriminator
from utils import *

class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        # self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_chn = config.g_chn
        self.ds_chn = config.ds_chn
        self.dt_chn = config.dt_chn
        self.n_frames = config.n_frames
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel
        self.gpus = config.gpus

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.n_class = config.n_class
        self.n_sample = config.n_sample
        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') TODO ADD
        self.device = torch.device('cpu') # just for test

        print("=" * 30)
        print('Build_model...')
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()


    def label_sampel(self):
        # label = torch.tensor(self.batch_size, dtype=torch.int64)
        label = torch.LongTensor(self.batch_size, 1).random_()%self.n_class
        one_hot= torch.zeros(self.batch_size, self.n_class).scatter_(1, label, 1)
        return label.squeeze(1).to(self.device), one_hot.to(self.device)       

    def train(self):

        # Data iterator

        # data_iter = iter(self.data_loader) TODO ADD
        # step_per_epoch = len(self.data_loader)

        # model_save_step = int(self.model_save_step * step_per_epoch) TODO ADD

        model_save_step = int(self.model_save_step * 10)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        print("=" * 30)
        print("Start training...")
        start_time = time.time()
        for step in range(start, self.total_step):

            batch_size = 5
            in_dim = 120
            n_frames = 4
            x = torch.randn(batch_size, in_dim)
            class_label = torch.randint(low=0, high=self.n_class, size=(batch_size,))
            # real_videos = torch.randn((batch_size, n_frames, 3, 64, 64)).cuda() TODO ADD
            real_videos = torch.randn((batch_size, n_frames, 3, 64, 64))


            self.D_s.train()
            self.D_t.train()
            self.G.train()

            # ================== Train D_s ================== #
            # try:
            #     real_images, real_labels = next(data_iter)
            # except:
            #     data_iter = iter(self.data_loader)
            #     real_images, real_labels = next(data_iter)

            # # Compute loss with real images

            # real_labels = real_labels.to(self.device)
            # real_images = real_images.to(self.device)

            ds_out_real = self.D_s(real_videos, class_label)
            if self.adv_loss == 'wgan-gp':
                ds_loss_real = - torch.mean(ds_out_real)
            elif self.adv_loss == 'hinge':
                ds_loss_real = torch.nn.ReLU()(1.0 - ds_out_real).mean()

            # apply Gumbel Softmax
            # z = torch.randn(self.batch_size, self.z_dim).to(self.device)

            # z_class, z_class_one_hot = self.label_sampel()

            fake_videos = self.G(x, class_label)

            fake_videos = sample_k_frames(fake_videos, len(fake_videos), self.n_sample)

            ds_out_fake = self.D_s(fake_videos.detach(), class_label)

            if self.adv_loss == 'wgan-gp':
                ds_loss_fake = ds_out_fake.mean()
            elif self.adv_loss == 'hinge':
                ds_loss_fake = torch.nn.ReLU()(1.0 + ds_out_fake).mean()


            # Backward + Optimize
            ds_loss = ds_loss_real + ds_loss_fake
            self.reset_grad()
            ds_loss.backward()
            self.ds_optimizer.step()


            # if self.adv_loss == 'wgan-gp':
            #     # Compute gradient penalty
            #     alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device).expand_as(real_images)
            #     interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
            #     out = self.D(interpolated)

            #     grad = torch.autograd.grad(outputs=out,
            #                                inputs=interpolated,
            #                                grad_outputs=torch.ones(out.size()).to(self.device),
            #                                retain_graph=True,
            #                                create_graph=True,
            #                                only_inputs=True)[0]

            #     grad = grad.view(grad.size(0), -1)
            #     grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            #     d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            #     # Backward + Optimize
            #     d_loss = self.lambda_gp * d_loss_gp

            #     self.reset_grad()
            #     d_loss.backward()
            #     self.d_optimizer.step()


            # ================== Train D_t ================== #
            # try:
            #     real_images, real_labels = next(data_iter)
            # except:
            #     data_iter = iter(self.data_loader)
            #     real_images, real_labels = next(data_iter)

            # # Compute loss with real images

            # real_labels = real_labels.to(self.device)
            # real_images = real_images.to(self.device)

            dt_out_real = self.D_t(real_videos, class_label)
            if self.adv_loss == 'wgan-gp':
                dt_loss_real = - torch.mean(dt_out_real)
            elif self.adv_loss == 'hinge':
                dt_loss_real = torch.nn.ReLU()(1.0 - dt_out_real).mean()

            # apply Gumbel Softmax
            # z = torch.randn(self.batch_size, self.z_dim).to(self.device)

            # z_class, z_class_one_hot = self.label_sampel()
 
            # fake_videos = self.G(z, class_label)
            d_out_fake = self.D_t(fake_videos.detach(), class_label)

            if self.adv_loss == 'wgan-gp':
                dt_loss_fake = dt_out_fake.mean()
            elif self.adv_loss == 'hinge':
                dt_loss_fake = torch.nn.ReLU()(1.0 + dt_out_fake).mean()


            # Backward + Optimize
            dt_loss = dt_loss_real + dt_loss_fake
            self.reset_grad()
            dt_loss.backward()
            self.dt_optimizer.step()


            # ================== Train G and gumbel ================== #
            # Create random noise
            # z = torch.randn(self.batch_size, self.z_dim).to(self.device)
            # z_class, z_class_one_hot = self.label_sampel()
            
            fake_videos = self.G(x, class_label)

            # Compute loss with fake images
            g_spatial_out_fake = self.D_s(fake_videos, class_label)  # Spatial Discrimminator loss
            g_temporal_out_fake = self.D_t(fake_videos, class_label) # Temporal Discriminator loss
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = -g_spatial_out_fake.mean() - g_temporal_out_fake.mean()
            # Same???
            elif self.adv_loss == 'hinge':
                g_loss_fake = -g_spatial_out_fake.mean() - g_temporal_out_fake.mean()

            self.reset_grad() # ?
            g_loss_fake.backward()
            self.g_optimizer.step()


            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_loss_fake: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_real.item(), d_loss_fake.item(), g_loss_fake.item()))
                
                if self.use_tensorboard:
                    self.writer.add_scalar('data/ds_loss_real', ds_loss_real.item(),(step + 1))
                    self.writer.add_scalar('data/ds_loss_fake', ds_loss_fake.item(),(step + 1))
                    self.writer.add_scalar('data/ds_loss', ds_loss.item(), (step + 1))

                    self.writer.add_scalar('data/dt_loss_real', dt_loss_real.item(),(step + 1))
                    self.writer.add_scalar('data/dt_loss_fake', dt_loss_fake.item(),(step + 1))
                    self.writer.add_scalar('data/dt_loss', dt_loss.item(), (step + 1))

                    self.writer.add_scalar('data/g_loss_fake', g_loss_fake.item(), (step + 1))


            # Sample images
            # Need to rewrite
            if (step + 1) % self.sample_step == 0:
                print('Sample images {}_fake.png'.format(step + 1))
                fake_images= self.G(fixed_z, z_class_one_hot)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D_s.state_dict(),
                           os.path.join(self.model_save_path, '{}_Ds.pth'.format(step + 1)))
                torch.save(self.D_t.state_dict(),
                           os.path.join(self.model_save_path, '{}_Dt.pth'.format(step + 1)))
            
            

    def build_model(self):

        self.G = Generator(self.z_dim, self.n_class, ch=self.g_chn, n_frames=self.n_frames).to(self.device)
        self.D_s = SpatialDiscriminator(chn=self.ds_chn, n_class=self.n_class).to(self.device)
        self.D_t = TemporalDiscriminator(chn=self.dt_chn, n_class=self.n_class).to(self.device)
        if self.parallel:
            print('use parallel...')
            print('gpuids ', self.gpus)
            gpus = [int(i) for i in self.gpus.split(',')]
    
            self.G = nn.DataParallel(self.G, device_ids=gpus)
            self.D_s = nn.DataParallel(self.D_s, device_ids=gpus)
            self.D_t = nn.DataParallel(self.D_t, device_ids=gpus)

        # self.G.apply(weights_init)
        # self.D.apply(weights_init)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.ds_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_s.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.dt_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_t.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()

        # print networks
        # print(self.G)
        # print(self.D_s)
        # print(self.D_t)

    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)
        
        tf_logs_path = os.path.join(self.log_path, 'tf_logs')
        self.writer = SummaryWriter(log_dir=tf_logs_path)


    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D_s.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Ds.pth'.format(self.pretrained_model))))
        self.D_t.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Dt.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.ds_optimizer.zero_grad()
        self.dt_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
