#!/bin/bash

if [[ $2 == "clean" ]]; then
  rm -rf logs samples models
fi

if [[ $1 == "local" ]]; then
  var="python3.6 main.py --adv_loss hinge --version biggan_lsun --parallel True --gpus 0 1 --num_workers 4\
  --use_tensorboard True --ds_chn 16 --dt_chn 16 --g_chn 16 --n_frames 8 --k_sample 4 --batch_size 8 \
  --n_class 1 \
  --root_path /home/haolin/Dataset/UCF101 \
  --annotation_path annotation/ucf101_1class_01.json \
  --log_path /home/haolin/Documents/logs \
  --model_save_path /home/haolin/Documents/models \
  --sample_path /home/haolin/Documents/samples"
  echo $var
  exec $var

elif [[ $1 == "remote" ]]; then
  var="python3.5 main.py --adv_loss hinge --version biggan_lsun --parallel True --gpus 4 5 6 7 --num_workers 12\
  --use_tensorboard True --ds_chn 16 --dt_chn 16 --g_chn 16 --n_frames 8 --k_sample 4 --batch_size 8 \
  --n_class 1 \
  --root_path /tmp4/potter/UCF101 \
  --annotation_path annotation/ucf101_1class_01.json \
  --log_path /tmp4/potter/outpus/logs \
  --model_save_path /tmp4/potter/outpus/models \
  --sample_path /tmp4/potter/outpus/samples"
  echo $var
  exec $var
fi
