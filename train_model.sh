#!/bin/bash

if [[ $1 == "local" ]]; then
  var="python3.6 main.py --adv_loss hinge --parallel True --gpus 0 1 --num_workers 4\
  --use_tensorboard True --ds_chn 16 --dt_chn 16 --g_chn 16 --n_frames 8 --k_sample 4 --batch_size 10 \
  --n_class 1 \
  --root_path /home/haolin/Dataset/UCF101 \
  --annotation_path annotation/ucf101_1class_01.json \
  --log_path /home/haolin/Documents/logs1 \
  --model_save_path /home/haolin/Documents/models1 \
  --sample_path /home/haolin/Documents/samples1"
  echo $var
  exec $var

elif [[ $1 == "vllab4" ]]; then
  var="python3.5 main.py --adv_loss hinge --parallel True --gpus 4 5 6 7 --num_workers 12\
  --use_tensorboard True --ds_chn 16 --dt_chn 16 --g_chn 16 --n_frames 8 --k_sample 4 --batch_size 20 \
  --n_class 1 \
  --root_path /tmp4/potter/UCF101 \
  --annotation_path annotation/ucf101_1class_01.json \
  --log_path /tmp4/potter/outputs/logs \
  --model_save_path /tmp4/potter/outputs/models \
  --sample_path /tmp4/potter/outputs/samples"
  echo $var
  exec $var
elif [[ $1 == "vllab2" ]]; then
  var="/home/potter/package/Python-3.5.2/python main.py --adv_loss wgan-gp --parallel True --gpus 1 2 3 --num_workers 16\
  --use_tensorboard True --ds_chn 16 --dt_chn 16 --g_chn 16 --n_frames 8 --k_sample 4 --batch_size 15 \
  --n_class 1 \
  --root_path /tmp4/potter/UCF101 \
  --annotation_path annotation/ucf101_1class_01.json \
  --log_path /tmp4/potter/outputs/logs \
  --model_save_path /tmp4/potter/outputs/models \
  --sample_path /tmp4/potter/outputs/samples\
  "
  echo $var
  exec $var
fi
