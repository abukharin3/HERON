# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym

from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--n_seeds", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--ensemble", type=str, default="linear",
        help="ensemble type")
    parser.add_argument("--alpha", type=float, default=1.0,
        help="the weight factor alpha")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HopperBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=50000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

if __name__ == "__main__":
    start = datetime.now()
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.n_seeds}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    exp_rewards = []
    for seed in range(args.n_seeds):
        # TRY NOT TO MODIFY: seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, seed, 0, args.capture_video, run_name)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        actor = Actor(envs).to(device)
        qf1 = QNetwork(envs).to(device)
        qf1_target = QNetwork(envs).to(device)
        target_actor = Actor(envs).to(device)
        target_actor.load_state_dict(actor.state_dict())
        qf1_target.load_state_dict(qf1.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

        actor2 = Actor(envs).to(device)
        qf2 = QNetwork(envs).to(device)
        qf2_target = QNetwork(envs).to(device)
        target_actor2 = Actor(envs).to(device)
        target_actor2.load_state_dict(actor2.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer2 = optim.Adam(list(qf2.parameters()), lr=args.learning_rate)
        actor_optimizer2 = optim.Adam(list(actor2.parameters()), lr=args.learning_rate)

        actor3 = Actor(envs).to(device)
        qf3 = QNetwork(envs).to(device)
        qf3_target = QNetwork(envs).to(device)
        target_actor3 = Actor(envs).to(device)
        target_actor3.load_state_dict(actor3.state_dict())
        qf3_target.load_state_dict(qf3.state_dict())
        q_optimizer3 = optim.Adam(list(qf3.parameters()), lr=args.learning_rate)
        actor_optimizer3 = optim.Adam(list(actor3.parameters()), lr=args.learning_rate)

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=True,
        )
        start_time = time.time()
        pendulum = PendulumEnv()
        # TRY NOT TO MODIFY: start the game
        obs = envs.reset()
        episode_returns = []
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions1 = actor(torch.Tensor(obs).to(device))
                    actions1 += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions1 = actions1.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

                    actions2 = actor2(torch.Tensor(obs).to(device))
                    actions2 += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions2 = actions2.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

                    actions3 = actor3(torch.Tensor(obs).to(device))
                    actions3 += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions3 = actions3.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

                    if args.ensemble == "linear":
                        actions = (actions1 + actions2 + actions3) / 3
                    elif args.ensemble == "preference":
                        w1 = args.alpha / (args.alpha + args.alpha ** 2 + args.alpha ** 3)
                        w2 = args.alpha ** 2 / (args.alpha + args.alpha ** 2 + args.alpha ** 3)
                        w3 = args.alpha ** 3 / (args.alpha + args.alpha ** 2 + args.alpha ** 3)
                        actions = w1 * actions1 + w2 * actions2 + w3 * actions3

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos= envs.step(actions)
            reward_info = np.array(envs.envs[0].get_reward_info(actions))

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    episode_returns.append(info['episode']['r'])
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    print(info["episode"]["r"])
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                th, thdot = data.observations[:, 0], data.observations[:, 1]

                r1, r2, r3 = -1 * angle_normalize(th) ** 2, -1 * thdot ** 2 , -1 * (data.actions ** 2)[:, 0]
                with torch.no_grad():
                    next_state_actions = target_actor(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    next_q_value = r1 + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                q_optimizer.zero_grad()
                qf1_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                with torch.no_grad():
                    next_state_actions = target_actor2(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    next_q_value2 = r2 + (1 - data.dones.flatten()) * args.gamma * (qf2_next_target).view(-1)

                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value2)

                # optimize the model
                q_optimizer2.zero_grad()
                qf2_loss.backward()
                q_optimizer2.step()

                if global_step % args.policy_frequency == 0:
                    actor_loss = -qf2(data.observations, actor2(data.observations)).mean()
                    actor_optimizer2.zero_grad()
                    actor_loss.backward()
                    actor_optimizer2.step()

                    # update the target network
                    for param, target_param in zip(actor2.parameters(), target_actor2.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                with torch.no_grad():
                    next_state_actions = target_actor3(data.next_observations)
                    qf3_next_target = qf3_target(data.next_observations, next_state_actions)
                    next_q_value3 = r3 + (1 - data.dones.flatten()) * args.gamma * (qf3_next_target).view(-1)

                qf3_a_values = qf3(data.observations, data.actions).view(-1)
                qf3_loss = F.mse_loss(qf3_a_values, next_q_value3)

                # optimize the model
                q_optimizer3.zero_grad()
                qf3_loss.backward()
                q_optimizer3.step()

                if global_step % args.policy_frequency == 0:
                    actor_loss = -qf3(data.observations, actor3(data.observations)).mean()
                    actor_optimizer3.zero_grad()
                    actor_loss.backward()
                    actor_optimizer3.step()

                    # update the target network
                    for param, target_param in zip(actor3.parameters(), target_actor3.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf3.parameters(), qf3_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        envs.close()
        writer.close()
        exp_rewards.append(episode_returns)
    exp_rewards = np.array(exp_rewards)
    end = datetime.now()
    print("Total time", end - start)
    #np.save("results_ensemble/ensemble_episode_returns_{}_{}_{}.npy".format(args.learning_rate, args.ensemble, args.alpha), exp_rewards)
