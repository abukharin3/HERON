#!/bin/sh


python cleanrl/rm_mountain.py --env-id MountainCarContinuous-v0 --learning-rate 1e-3 --n_seeds 3 --sigma_mult 1.0
python cleanrl/ddpg_mountain_baseline.py --env-id MountainCarContinuous-v0 --learning-rate 1e-3 --n_seeds 3
