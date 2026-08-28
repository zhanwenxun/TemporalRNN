#!/bin/bash

python rq3.py -dataset "mnist" -model_type "lstm"&
python rq3.py -dataset "mnist" -model_type "gru" 
wait

python rq3.py -dataset "mnist" -model_type "blstm" &
python rq3.py -dataset "fashion" -model_type "lstm"
wait

python rq3.py -dataset "fashion" -model_type "gru" &
python rq3.py -dataset "fashion" -model_type "blstm"
wait

python rq3.py -dataset "svhn" -model_type "lstm" &
python rq3.py -dataset "svhn" -model_type "gru"
wait

python rq3.py -dataset "svhn" -model_type "blstm"
wait

python rq3.py -dataset "agnews" -model_type "lstm" &
python rq3.py -dataset "agnews" -model_type "gru"
wait 

python rq3.py -dataset "agnews" -model_type "blstm" &
python rq3.py -dataset "snips" -model_type "lstm"
wait

python rq3.py -dataset "snips" -model_type "gru" &
python rq3.py -dataset "snips" -model_type "blstm"
