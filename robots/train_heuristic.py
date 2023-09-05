import os, argparse, random
from pybulletgym.envs.mujoco.envs.locomotion.half_cheetah_env import HalfCheetahMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.ant_env import AntMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.hopper_env import HopperMuJoCoEnv
# Patch and register pybullet envs
# import rl_zoo3.gym_patches
# import pybullet_envs

from stable_baselines3 import PPO
from stable_baselines3.ppo.heron import HERON
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import torch

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", choices=["hopper", "ant", "cheetah"], default="cheetah")
	parser.add_argument("--time_steps", type=int, default=1000)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--order", type=str, default="0,1,2,3")
	parser.add_argument("--alpha", type=float, default=0.5)
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)


	if args.env == "hopper":
		order = [int(x) for x in args.order.split(",")][:3]
		env = HopperMuJoCoEnv()
		factor_dim=3
	elif args.env == "ant":
		order = [int(x) for x in args.order.split(",")]
		env = AntMuJoCoEnv()
		factor_dim=4
	else:
		order = [int(x) for x in args.order.split(",")][:2]
		factor_dim=2
		env = HalfCheetahMuJoCoEnv()

		
	env.reset()
	env = DummyVecEnv([lambda: env])
	env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

	model = HERON("MlpPolicy", env, verbose=True, learning_rate=3e-4, heirarchy=order, factor_dim=factor_dim, multiple_sigmas=False, heron=False, heuristic=True, alpha=args.alpha, sigma=0)

	results = model.learn(total_timesteps=args.time_steps)
	print(np.array(results.stored_rewards))
	order = "_".join(args.order.split(","))

	np.save(f"results/heuristic_main_{args.env}_{args.time_steps}_{order}_{args.alpha}_{args.seed}.npy", np.array(results.stored_rewards))

	model.save(f"results/heuristic_main_{args.env}_{args.time_steps}_{order}_{args.alpha}_{args.seed}")
	if not os.path.exists(f"results/heuristic_main_{args.env}_{args.time_steps}_{order}_{args.alpha}_{args.seed}"):
		os.mkdir(f"results/heuristic_main_{args.env}_{args.time_steps}_{order}_{args.alpha}_{args.seed}")
	stats_path = os.path.join(f"results/heuristic_main_{args.env}_{args.time_steps}_{order}_{args.alpha}_{args.seed}.pkl")#, "vec_normalize.pkl")
	env.save(stats_path)

	# env = HopperMuJoCoEnv(render=True)
	# env = DummyVecEnv([lambda: env])
	# env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
	# mean_reward, std_reward = evaluate_policy(model, env)

	# Don't forget to save the VecNormalize statistics when saving the agent
	