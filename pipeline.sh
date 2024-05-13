#!/usr/bin/env bash
set -e
CONFIG1=car-in-1.txt
CONFIG2=car-in-2.txt


python run_nerf.py --config $CONFIG1 --ft_path './logs/car/200000.tar' --step1_initialize
python run_nerf.py --config $CONFIG2 --ft_path './logs/car-in/250000.tar' --step2_freeze
python run_nerf.py --config $CONFIG2 --ft_path './logs/car-in/450000.tar' --step3_unfreeze
