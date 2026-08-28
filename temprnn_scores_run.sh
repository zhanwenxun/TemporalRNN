#!/bin/bash
# 为什么对于不同的数据集设置不同的超参数，是因为rnn_scores模型对于不同数据集预测结果开始稳定的时间步数不一样

python temprnn_scores.py -dataset "mnist" -model "lstm" -prostart 5 -labelstart 5
python temprnn_scores.py -dataset "mnist" -model "gru" -prostart 5 -labelstart 5
python temprnn_scores.py -dataset "mnist" -model "blstm" -prostart 5 -labelstart 5

python temprnn_scores.py -dataset "fashion" -model "lstm" -prostart 5 -labelstart 0
python temprnn_scores.py -dataset "fashion" -model "gru" -prostart 5 -labelstart 0
python temprnn_scores.py -dataset "fashion" -model "blstm" -prostart 5 -labelstart 0

python temprnn_scores.py -dataset "svhn" -model "lstm" -prostart 10 -labelstart 0
python temprnn_scores.py -dataset "svhn" -model "gru" -prostart 10 -labelstart 0
python temprnn_scores.py -dataset "svhn" -model "blstm" -prostart 10 -labelstart 0

python temprnn_scores.py -dataset "agnews" -model "lstm" -prostart 20 -labelstart 0
python temprnn_scores.py -dataset "agnews" -model "gru" -prostart 20 -labelstart 0
python temprnn_scores.py -dataset "agnews" -model "blstm" -prostart 20 -labelstart 0

python temprnn_scores.py -dataset "snips" -model "lstm" -prostart 10 -labelstart 0
python temprnn_scores.py -dataset "snips" -model "gru" -prostart 10 -labelstart 0
python temprnn_scores.py -dataset "snips" -model "blstm" -prostart 10 -labelstart 0

wait