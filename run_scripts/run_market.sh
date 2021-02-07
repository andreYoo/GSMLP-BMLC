#!/bin/bash
#python tools/train.py --experiments experiments/market.yml --gpus 0,1 --suffix 'market_test_v5'
python tools/train.py --experiments experiments/market2.yml --gpus 0,1 --suffix 'market_test_v6'
python tools/train.py --experiments experiments/market3.yml --gpus 0,1 --suffix 'market_test_v7'
python tools/train.py --experiments experiments/market4.yml --gpus 0,1 --suffix 'market_test_v8'
#python tools/train.py --experiments experiments/market_v2.yml --gpus 0,1
#python tools/train.py --experiments experiments/market_v3.yml --gpus 1
