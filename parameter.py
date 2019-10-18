import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='dvd-gan', choices=['dvd-gan'])
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--imsize', type=int, default=128)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--g_chn', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=120)
    parser.add_argument('--ds_chn', type=int, default=32) # SpatialDiscriminator channel
    parser.add_argument('--dt_chn', type=int, default=32) # TemporalDiscriminator channel
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lr_schr', type=str, default='const', choices=['const', 'step', 'exp', 'multi', 'reduce'])
    parser.add_argument('--version', type=str, default='')

    # Training setting
    parser.add_argument('--total_epoch', type=int, default=100000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=int, default=1)
    parser.add_argument('--g_iters', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--g_lr', type=float, default=5e-5)
    parser.add_argument('--d_lr', type=float, default=5e-5)
    parser.add_argument('--lr_decay', type=float, default=0.9999)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('-g', '--gpus', default=[], nargs='+', type=str, help='Specify GPU ids.')
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'kinetics','activitynet', 'hmdb51'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--n_class', type=int, default=101)
    parser.add_argument('--k_sample', type=int, default=64)
    parser.add_argument('--n_frames', type=int, default=24)
    parser.add_argument('--test_batch_size', type=int, default=8, help='how many batchsize for test and sample')

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')

    # epoch size
    parser.add_argument('--log_epoch', type=int, default=1)
    parser.add_argument('--sample_epoch', type=int, default=20)
    parser.add_argument('--model_save_epoch', type=int, default=200)

    # Dataloader
    parser.add_argument('--norm_value', type=int, default=255)
    parser.add_argument('--no_mean_norm', action='store_true', default=True)
    parser.add_argument('--std_norm', action='store_true', default=False)
    parser.add_argument('--mean_dataset', type=str, default='activitynet')
    parser.add_argument('--root_path', type=str, default='/tmp4/potter/UCF101')
    parser.add_argument('--video_path', type=str, default='videos_jpeg')
    parser.add_argument('--annotation_path', type=str, default='annotation/ucf101_01.json')
    parser.add_argument('--train_crop', type=str, default='corner') #corner | random | center
    parser.add_argument('--sample_size', type=int, default=64)

    parser.add_argument('--initial_scale', type=float, default=1.0) 
    parser.add_argument('--n_scales', type=int, default=5)
    parser.add_argument('--scale_step', type=float, default=0.84089641525)

    config = parser.parse_args()

    return config
