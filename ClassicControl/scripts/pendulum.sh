#!/bin/sh

python cleanrl/ddpg_rm.py --env-id Pendulum-v1 --learning-rate 5e-4 --rm-learning-rate 1e-4 --n_seeds 3
python cleanrl/ddpg_continuous_action.py --env-id Pendulum-v1 --learning-rate 5e-4 --n_seeds 3
