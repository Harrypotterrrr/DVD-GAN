#!/bin/bash

DATASET_PATH=$2

if [[ $1 == "local" ]]; then
  var="python3.6 main.py --adv_loss hinge --parallel True --gpus 0 1 --num_workers 8 \
  --use_tensorboard True --ds_chn 32 --dt_chn 32 --g_chn 32 --n_frames 8 --k_sample 4 --batch_size 32 \
  --g_lr 5e-4 \
  --d_lr 5e-5 \
  --n_class 1 \
  --lr_schr const \
  --n_class 2
  --root_path /home/haolin/Dataset/UCF101 \
  --annotation_path annotation/ucf101_01.json \
  --log_path /home/haolin/Documents/outputs/logs \
  --model_save_path /home/haolin/Documents/outputs/models \
  --sample_path /home/haolin/Documents/outputs/samples \
  "
  echo $var
  exec $var

elif [[ $1 == "vllab4" ]]; then
  var="python3.5 main.py --adv_loss hinge --parallel True --gpus 3 7 --num_workers 12 \
  --use_tensorboard True --ds_chn 64 --dt_chn 64 --g_chn 64 --n_frames 8 --k_sample 4 --batch_size 8 \
  --n_class 1 \
  --root_path /tmp4/potter/UCF101 \
  --annotation_path annotation/ucf101_1class_01.json \
  --log_path /tmp4/potter/outputs/logs \
  --model_save_path /tmp4/potter/outputs/models \
  --sample_path /tmp4/potter/outputs/samples \
  "
  echo $var
  exec $var

elif [[ ($1 == "vllab2") || ($1 == "vllab3") ]]; then
  var="/home/potter/package/Python-3.5.2/python main.py --adv_loss hinge --parallel True --gpus 2 3 --num_workers 16 \
  --use_tensorboard True --ds_chn 64 --dt_chn 64 --g_chn 64 --n_frames 8 --k_sample 4 --batch_size 6 \
  --n_class 2 \
  --root_path /tmp3/potter/UCF101 \
  --annotation_path annotation/ucf101_01.json \
  --log_path /tmp3/potter/outputs/logs \
  --model_save_path /tmp3/potter/outputs/models \
  --sample_path /tmp3/potter/outputs/samples \
  --g_lr 4e-5 --d_lr 4e-5 \
  --lr_schr multi
  "
  echo $var
  exec $var

elif [[ $1 == "GCP" ]]; then
  var="python3.6 main.py --adv_loss hinge --parallel True --gpus 0 1 2 3 --num_workers 16 \
  --use_tensorboard True --ds_chn 64 --dt_chn 64 --g_chn 64 --n_frames 8 --k_sample 4 --batch_size 40 \
  --n_class 1 \
  --root_path $2/UCF101 \
  --annotation_path annotations/ucf101_01.json \
  --log_path ~/outputs/logs \
  --model_save_path ~/outputs/models \
  --sample_path ~/outputs/samples \
  "
  echo $var
  exec $var

fi
