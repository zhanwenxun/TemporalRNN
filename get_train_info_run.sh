#!/bin/bash

# 考虑下要不要直接把RNN模型的输出直接处理成7个特征，省的还需要保存大容量的信息

# 使用全部训练数据，跑出训练集的预测信息
python get_train_info.py -dataset 'mnist' -model_type 'lstm' -sample_size 'all'
python get_train_info.py -dataset 'mnist' -model_type 'gru' -sample_size 'all'
python get_train_info.py -dataset 'mnist' -model_type 'blstm' -sample_size 'all'
echo "mnist 数据集完成"

python get_train_info.py -dataset 'fashion' -model_type 'lstm' -sample_size 'all'
python get_train_info.py -dataset 'fashion' -model_type 'gru' -sample_size 'all'
python get_train_info.py -dataset 'fashion' -model_type 'blstm' -sample_size 'all'
echo "fashion 数据集完成"

python get_train_info.py -dataset 'svhn' -model_type 'lstm' -sample_size 'all'
python get_train_info.py -dataset 'svhn' -model_type 'gru' -sample_size 'all'
python get_train_info.py -dataset 'svhn' -model_type 'blstm' -sample_size 'all'
echo "svhn 数据集完成"

python get_train_info.py -dataset 'agnews' -model_type 'lstm' -sample_size 'all'
python get_train_info.py -dataset 'agnews' -model_type 'gru' -sample_size 'all'
python get_train_info.py -dataset 'agnews' -model_type 'blstm' -sample_size 'all'
echo "agnews 数据集完成"

python get_train_info.py -dataset 'snips' -model_type 'lstm' -sample_size 'all'
python get_train_info.py -dataset 'snips' -model_type 'gru' -sample_size 'all'
python get_train_info.py -dataset 'snips' -model_type 'blstm' -sample_size 'all'
echo "snips 数据集完成"