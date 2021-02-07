#!/bin/bash
python tools/train.py --experiments experiments/msmt17_v1.yml --gpus 0,1 --suffix 'msmt_17_v1'
#python tools/train.py --experiments experiments/msmt17_v2.yml --gpus 0,1 --suffix 'msmt_17_v2'
#python tools/train.py --experiments experiments/msmt17_v3.yml --gpus 0,1 --suffix 'msmt_17_v3'
