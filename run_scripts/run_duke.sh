#!/bin/bash
#python tools/train.py --experiments experiments/duke_mlp.yml --gpus 0,1 --suffix 'testi_smlc' --mlp 'SMLC'
#python tools/train.py --experiments experiments/duke_mlp.yml --gpus 0,1 --suffix 'test_ss' --mlp 'SS'
python tools/train.py --experiments experiments/duke_mlp.yml --gpus 0,1 --suffix 'test_knn' --mlp 'KNN'
