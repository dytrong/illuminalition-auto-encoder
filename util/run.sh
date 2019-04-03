#! /bin/zsh 
EPOCH=500
optimizer_method='Adam'
model_parameters_path="../parameters/i_4608_autoencoder_cnn_${optimizer_method}_${EPOCH}.pth"


(CUDA_VISIBLE_DEVICES=0  python -u train_cnn_autoencoder.py \
--EPOCH ${EPOCH} \
--optimizer_method ${optimizer_method} \
--model_parameters_path ${model_parameters_path} > "./log/i_4608_autoencoder_cnn_${optimizer_method}_${EPOCH}.txt"
)
