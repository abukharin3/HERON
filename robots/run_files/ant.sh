#!/bin/sh

python train_heron.py --env ant --time_steps 2000000 --seed 0 --order 0,2,1,3 --sigma 0.1 --rlhf
python train_heron.py --env ant --time_steps 2000000 --seed 0 --order 0,2,1,3 --sigma 0.1 --rlhf
python train_heron.py --env ant --time_steps 2000000 --seed 0 --order 0,2,1,3 --sigma 0.1 --rlhf

python train_heron.py --env ant --time_steps 2000000 --seed 0 --order 0,2,1,3 --sigma 1.0,1.0,0.5,0.5
python train_heron.py --env ant --time_steps 2000000 --seed 1 --order 0,2,1,3 --sigma 1.0,1.0,0.5,0.5
python train_heron.py --env ant --time_steps 2000000 --seed 2 --order 0,2,1,3 --sigma 1.0,1.0,0.5,0.5
