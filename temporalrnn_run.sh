#!/bin/bash
set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# cd "${PROJECT_ROOT}"

OUTPUT_DIR="./results/temporalrnn"
COMMON_ARGS="--budgets 0.1 0.2 --file-count 30 --skip-missing --output-dir ${OUTPUT_DIR}"
METHOD_ARGS="--score-multiplier 2.0 --margin-multiplier 2.0 --min-group-quota 1 --seed 42"

python temporalrnn.py -dataset "mnist" -model "lstm" -prostart 5 -labelstart 5 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "mnist" -model "gru" -prostart 5 -labelstart 5 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "mnist" -model "blstm" -prostart 5 -labelstart 5 ${COMMON_ARGS} ${METHOD_ARGS}

python temporalrnn.py -dataset "fashion" -model "lstm" -prostart 5 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "fashion" -model "gru" -prostart 5 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "fashion" -model "blstm" -prostart 5 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}

python temporalrnn.py -dataset "svhn" -model "lstm" -prostart 10 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "svhn" -model "gru" -prostart 10 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "svhn" -model "blstm" -prostart 10 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}

python temporalrnn.py -dataset "agnews" -model "lstm" -prostart 20 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "agnews" -model "gru" -prostart 20 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "agnews" -model "blstm" -prostart 20 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}

python temporalrnn.py -dataset "snips" -model "lstm" -prostart 10 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "snips" -model "gru" -prostart 10 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}
python temporalrnn.py -dataset "snips" -model "blstm" -prostart 10 -labelstart 0 ${COMMON_ARGS} ${METHOD_ARGS}

python temporalrnn.py --aggregate-existing --output-dir "${OUTPUT_DIR}"
