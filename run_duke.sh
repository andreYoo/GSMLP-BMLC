#!/bin/bash
python tools/train.py --experiments experiments/duke.yml --gpus 0,1 --suffix 'experiment' --mlp 'SMLC'
