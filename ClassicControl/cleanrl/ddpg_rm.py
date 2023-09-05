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
    parser.add_argument("--sigma_mult", type=float, default=1.0,
        help="what to multiply std by")

    # RM parameters
    parser.add_argument("--rm-learning-rate", type=float, default=1e-3,
        help="learning rate of RK")
    parser.add_argument("--rm-normalize", action='store_true', default=False,
        help="learning rate of RK")
    parser.add_argument("--heuristic", action='store_true', default=False,
        help="learning rate of RK")
    parser.add_argument("--beta", type=float, default=0.5,
        help="beta in heuristic")
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


# ALGO LOGIC: initialize agent here:
class RewardModel(nn.Module):
    def __init__(self, env, lr, normalize=False):
        super().__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.mu = 0
        self.sigma = 1
        self.normalize = normalize

        self.heirarchy = [0, 1, 2]

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.normalize:
            x = (x - self.mu) / self.sigma
        return x

    def get_loss(self, x, reward_signal, sigmas):
        sign = [-1, -1, -1]
        total_loss = 0
        total = 0
        correct = 0
        outs = []
        for i in range(x.shape[0]):
            for j in range(i):
                x_i = x[i]
                x_j = x[j]

                reward_i = self(x_i)
                if j == 0:
                    outs.append(reward_i.item())
                reward_j = self(x_j)

                reward_info_i = reward_signal[i]
                reward_info_j = reward_signal[j]


                # Level 1
                if reward_info_i[self.heirarchy[0]] * sign[0] > reward_info_j[self.heirarchy[0]] * sign[0] + sigmas[self.heirarchy[0]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_i - reward_j))
                    if reward_i > reward_j:
                        correct += 1
                elif reward_info_j[self.heirarchy[0]] * sign[0] > reward_info_i[self.heirarchy[0]] * sign[0] + sigmas[self.heirarchy[0]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_j - reward_i))
                    if reward_j > reward_i:
                        correct += 1
                # Level 2
                elif reward_info_i[self.heirarchy[1]] * sign[1] > reward_info_j[self.heirarchy[1]] * sign[1] + sigmas[self.heirarchy[1]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_i - reward_j))
                    if reward_i > reward_j:
                        correct += 1
                elif reward_info_j[self.heirarchy[1]] * sign[1] > reward_info_i[self.heirarchy[1]] * sign[1] + sigmas[self.heirarchy[1]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_j - reward_i))
                    if reward_j > reward_i:
                        correct += 1
                # Level 3
                elif reward_info_i[self.heirarchy[2]] * sign[2] > reward_info_j[self.heirarchy[2]] * sign[2] + sigmas[self.heirarchy[2]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_i - reward_j))
                    if reward_i > reward_j:
                        correct += 1
                elif reward_info_j[self.heirarchy[2]] * sign[2] > reward_info_i[self.heirarchy[2]] * sign[2] + sigmas[self.heirarchy[2]]:
                    loss = -1 * torch.log(torch.sigmoid(reward_j - reward_i))
                    if reward_j > reward_i:
                        correct += 1
                else:
                    continue
                total += 1
                total_loss += loss
        return total_loss / (total + 1e-5), correct / (total + 1e-5), outs





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

def train_rm(rm, state_buffer, reward_buffer, bsz = 16, n_batch=16, sigma_mult=1):
    state_buffer = torch.Tensor(np.stack(state_buffer))
    reward_buffer = torch.Tensor(np.stack(reward_buffer))
    sigmas = torch.Tensor(reward_buffer.std(0)) * sigma_mult
    total_loss = 0
    total_acc = 0
    total = 0
    reward_scale = []
    for i in range(n_batch):
        idx = np.random.choice(state_buffer.shape[0], bsz)
        loss, acc, outs = rm.get_loss(state_buffer[idx], reward_buffer[idx], sigmas)
        if loss <= 0:
            continue
        reward_scale += outs
        rm.optimizer.zero_grad()
        loss.backward()
        rm.optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        total += 1

    rm.mu, rm.sigma = np.array(reward_scale).mean(), np.array(reward_scale).std()

    return total_loss / (total+1e-5), total_acc / (total+1e-5)

def get_stats(rm, state_buffer, reward_buffer, bsz = 16, n_batch=16, sigma_mult=1):
    state_buffer = torch.Tensor(np.stack(state_buffer))
    reward_buffer = torch.Tensor(np.stack(reward_buffer))
    sigmas = torch.Tensor(reward_buffer.std(0)) * sigma_mult
    reward_scale = []
    for i in range(n_batch):
        idx = np.random.choice(reward_buffer.shape[0], bsz)
        reward_scale.append(reward_buffer[idx].numpy())

    rm.mu, rm.sigma = np.concatenate(reward_scale, axis=0).mean(0), np.concatenate(reward_scale, axis=0).std(0)

    return rm.mu, rm.sigma




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

        device = "cpu"#torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, seed, 0, args.capture_video, run_name)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        actor = Actor(envs).to(device)
        RM = RewardModel(envs, lr=args.rm_learning_rate, normalize=args.rm_normalize).to(device)
        qf1 = QNetwork(envs).to(device)
        qf1_target = QNetwork(envs).to(device)
        target_actor = Actor(envs).to(device)
        target_actor.load_state_dict(actor.state_dict())
        qf1_target.load_state_dict(qf1.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

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
        state_buffer = []
        signal_buffer = []
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(device))
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos= envs.step(actions)

            # Get reward information
            reward_info = np.array(envs.envs[0].get_reward_info(actions))
            #print(reward_info[0] + reward_info[1] * 0.1 + reward_info[2] * 0.001, rewards)
            
            state_buffer.append(np.concatenate([np.squeeze(obs), reward_info]))
            if  global_step > 1000 and args.heuristic:
                mu, sigma  = get_stats(RM, state_buffer, signal_buffer, sigma_mult=args.sigma_mult)
            signal_buffer.append(reward_info)

            if global_step > 1000 and global_step % 400 == 0 and not args.heuristic:
                loss, acc = train_rm(RM, state_buffer, signal_buffer, sigma_mult=args.sigma_mult)
                print(global_step, loss, acc)

            rewards = RM(torch.Tensor(np.concatenate([np.squeeze(obs), reward_info]))).detach().numpy()
            if args.heuristic and global_step > 1000:
                rewards = -1 * reward_info[0] - args.beta ** 1 * reward_info[1] - args.beta ** 2 * reward_info[2]
                #rewards = -args.beta * (reward_info[0] -mu[0]) / sigma[0] - args.beta ** 2 * (reward_info[1] -mu[1]) / sigma[1] - args.beta ** 3 * (reward_info[2] -mu[2]) / sigma[2]
                
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
                with torch.no_grad():
                    next_state_actions = target_actor(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

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
    np.save("results/rm_episode_returns_{}_{}_{}_{}_{}_{}.npy".format(args.learning_rate, args.rm_learning_rate, args.rm_normalize, args.sigma_mult, args.heuristic, args.beta), exp_rewards)
