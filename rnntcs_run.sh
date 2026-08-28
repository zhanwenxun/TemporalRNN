#!/bin/bash

# python rnntcs.py -dataset "mnist" -model_type "lstm" &
# python rnntcs.py -dataset "mnist" -model_type "gru" &
# python rnntcs.py -dataset "mnist" -model_type "blstm"
# wait

# python rnntcs.py -dataset "fashion" -model_type "lstm" &
# python rnntcs.py -dataset "fashion" -model_type "gru" &
# python rnntcs.py -dataset "fashion" -model_type "blstm"
# wait

# python rnntcs.py -dataset "svhn" -model_type "lstm" &
# python rnntcs.py -dataset "svhn" -model_type "gru" &
# python rnntcs.py -dataset "svhn" -model_type "blstm"
# wait

python rnntcs.py -dataset "agnews" -model_type "lstm" &
python rnntcs.py -dataset "agnews" -model_type "gru" &
python rnntcs.py -dataset "agnews" -model_type "blstm"
wait

python rnntcs.py -dataset "snips" -model_type "lstm" &
python rnntcs.py -dataset "snips" -model_type "gru" &
python rnntcs.py -dataset "snips" -model_type "blstm"
wait