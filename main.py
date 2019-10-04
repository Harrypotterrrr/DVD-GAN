from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
from parameter import get_parameters

import os
import torch


##### Import libary for dataloader #####
##### https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/main.py
from Dataloader.transform.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from Dataloader.transform.temporal_transforms import LoopPadding, TemporalRandomCrop
from Dataloader.transform.target_transforms import ClassLabel, VideoID
from Dataloader.transform.target_transforms import Compose as TargetCompose
from Dataloader.dataloader import get_training_set, get_validation_set, get_test_set
from Dataloader.mean import get_mean


def main(config):
    # For fast training
    cudnn.benchmark = True

    ##### Dataloader #####
    config.video_path = os.path.join(config.root_path, config.video_path)
    config.annotation_path = os.path.join(config.root_path, config.annotation_path)
    config.mean = get_mean(config.norm_value, dataset=config.mean_dataset)

    if config.no_mean_norm and not config.std_norm:
        norm_method = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif not config.std_norm:
        norm_method = Normalize(config.mean, [1, 1, 1])

    config.scales = [config.initial_scale]
    for i in range(1, config.n_scales):
        config.scales.append(config.scales[-1] * config.scale_step)

    if config.train:
        assert config.train_crop in ['random', 'corner', 'center']
        if config.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(config.scales, config.sample_size)
        elif config.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(config.scales, config.sample_size)
        elif config.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                config.scales, config.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(config.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(config.n_frames)
        target_transform = ClassLabel()

        print("="*30,"\nLoading data...")
        training_data = get_training_set(config, spatial_transform,
                                         temporal_transform, target_transform)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True)
    else:
        spatial_transform = Compose([
            Scale(config.sample_size),
            CenterCrop(config.sample_size),
            ToTensor(config.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(config.n_frames)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            config, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True)

    ##### End dataloader #####

    # Use Big-GAN dataset to test only
    # The random data is used in the trainer
    # Need to pre-process data and use the dataloader (above)

    # config.n_class = len(glob.glob(os.path.join(config.root_path, config.video_path)))

    ## Data loader
    print('number class:', config.n_class)
    # # Data loader
    # data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize,
    #                          config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    # make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)

    if config.train:
        if config.model=='dvd-gan':
            trainer = Trainer(train_loader, config) 
        else:
            trainer = None

        trainer.train()
    else:
        tester = Tester(val_loader, config)
        tester.test()


if __name__ == '__main__':
    config = get_parameters()

    for key in config.__dict__.keys():
        print(key, "=", config.__dict__[key])

    main(config)