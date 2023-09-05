import os, argparse
from pybulletgym.envs.mujoco.envs.locomotion.half_cheetah_env import HalfCheetahMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.ant_env import AntMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.hopper_env import HopperMuJoCoEnv
# Patch and register pybullet envs
# import rl_zoo3.gym_patches
# import pybullet_envs

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", choices=["hopper", "ant", "cheetah"], default="ant")
	args = parser.parse_args()

	if args.env == "hopper":
		env = HopperMuJoCoEnv(render=True)
	elif args.env == "ant":
		env = AntMuJoCoEnv(render=True)
	else:
		env = HalfCheetahMuJoCoEnv(render=True)
		
	env = DummyVecEnv([lambda: env])

	env = VecNormalize.load("results/baseline_results/ppo_ant_2000000/vec_normalize.pkl", env)

	model = PPO.load("results/baseline_results/ppo_ant_2000000.zip")
	mean_reward, std_reward = evaluate_policy(model, env)

	# results = model.learn(total_timesteps=args.time_steps)
	# np.save(f"results/{args.env}_{args.time_steps}.npy", np.array(results.stored_rewards))

	# model.save(f"results/ppo_{args.env}_{args.time_steps}")
	# if not os.path.exists("results/ppo_{args.env}_{args.time_steps}"):
	# 	os.mkdir("results/ppo_{args.env}_{args.time_steps}")
	# stats_path = os.path.join("results/ppo_{args.env}_{args.time_steps}", "vec_normalize.pkl")
	# env.save(stats_path)

	# env = HopperMuJoCoEnv(render=True)
	# env = DummyVecEnv([lambda: env])
	# env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
	# mean_reward, std_reward = evaluate_policy(model, env)

	# Don't forget to save the VecNormalize statistics when saving the agent
	