#!/bin/bash

# 第一组：mnist
python rq12.py -dataset "mnist" -model_type "lstm" &
python rq12.py -dataset "mnist" -model_type "gru" &
python rq12.py -dataset "mnist" -model_type "blstm" &
wait
echo "mnist 数据集完成"

# 第二组：fashion
python rq12.py -dataset "fashion" -model_type "lstm" &
python rq12.py -dataset "fashion" -model_type "gru" &
python rq12.py -dataset "fashion" -model_type "blstm" 
wait
echo "fashion 数据集完成"

# 第三组：svhn
python rq12.py -dataset "svhn" -model_type "lstm" &
python rq12.py -dataset "svhn" -model_type "gru" 
python rq12.py -dataset "svhn" -model_type "blstm" &
wait
echo "svhn 数据集完成"

# 第四组：agnews
python rq12.py -dataset "agnews" -model_type "lstm" &
python rq12.py -dataset "agnews" -model_type "gru" 
python rq12.py -dataset "agnews" -model_type "blstm" &
wait
echo "agnews 数据集完成"

# 第五组：snips
python rq12.py -dataset "snips" -model_type "lstm" 
python rq12.py -dataset "snips" -model_type "gru" &
python rq12.py -dataset "snips" -model_type "blstm" 
wait
echo "snips 数据集完成"