#!/bin/bash

cd RNNModels/mnist_demo
python mnist_lstm.py -type "train"
python mnist_gru.py -type "train"
python mnist_blstm.py -type "train"

cd ../fashion_demo
python fashion_lstm.py -type "train"
python fashion_gru.py -type "train"
python fashion_blstm.py -type "train"

cd ../svhn_demo
python svhn_lstm.py -type "train"
python svhn_gru.py -type "train"
python svhn_blstm.py -type "train"

cd ../agnews_demo
python agnews_lstm.py -type "train"
python agnews_gru.py -type "train"
python agnews_blstm.py -type "train"

cd ../snips_demo
python snips_lstm.py -type "train"
python snips_gru.py -type "train"
python snips_blstm.py -type "train"

cd ../../gen_data/gen_test_dataset
python dau_mnist.py
python dau_fashion.py
python dau_svhn.py
# python dau_agnews.py
# python dau_snips.py

# for RQ1 & RQ2
python gen_toselect_dataset.py -dataset "mnist"
python gen_toselect_dataset.py -dataset "fashion"
python gen_toselect_dataset.py -dataset "svhn"
python gen_toselect_dataset.py -dataset "agnews"
python gen_toselect_dataset.py -dataset "snips"

# DeepStellar
cd ../..
python ./abstraction_runner.py -test_obj "mnist_lstm"
python ./abstraction_runner.py -test_obj "mnist_blstm"
python ./abstraction_runner.py -test_obj "mnist_gru"

python ./abstraction_runner.py -test_obj "fashion_lstm"
python ./abstraction_runner.py -test_obj "fashion_blstm"
python ./abstraction_runner.py -test_obj "fashion_gru"

python ./abstraction_runner.py -test_obj "svhn_lstm"
python ./abstraction_runner.py -test_obj "svhn_blstm"
python ./abstraction_runner.py -test_obj "svhn_gru"

python ./abstraction_runner.py -test_obj "agnews_lstm"
python ./abstraction_runner.py -test_obj "agnews_blstm"
python ./abstraction_runner.py -test_obj "agnews_gru"

python ./abstraction_runner.py -test_obj "snips_lstm"
python ./abstraction_runner.py -test_obj "snips_blstm"
python ./abstraction_runner.py -test_obj "snips_gru"