#!/bin/sh

python train.py --env cheetah --time_steps 1000000 --seed 0
python train.py --env cheetah --time_steps 1000000 --seed 1
python train.py --env cheetah --time_steps 1000000 --seed 2

python train_heron.py --env cheetah --time_steps 1000000 --seed 0 --order 0,1 --sigma 0.5,0.1
python train_heron.py --env cheetah --time_steps 1000000 --seed 1 --order 0,1 --sigma 0.5,0.1
python train_heron.py --env cheetah --time_steps 1000000 --seed 2 --order 0,1 --sigma 0.5,0.1