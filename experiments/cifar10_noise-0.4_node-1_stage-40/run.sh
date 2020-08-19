#!/usr/bin/env bash
work_path=$(dirname $0)
python -u main.py --algo ddpg --config $work_path"/cifar10_noise_config.yaml"
 #--load-path=$work_path/ckpt.pth.tar \
 #--recover

