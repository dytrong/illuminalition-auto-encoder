#!/bin/zsh

dataset='i'
model_parameters_path="../parameters/i_4608_autoencoder_cnn_Adam_500.pth"


(CUDA_VISIBLE_DEVICES=1  python -u run.py \
--dataset ${dataset} \
--model_parameters_path ${model_parameters_path} > "./log/${dataset}_4608_autoencoder_cnn_Adam_500.txt")

echo "日志保存地址为: ./log/${dataset}_4608_autoencoder_cnn_Adam_500.txt"



