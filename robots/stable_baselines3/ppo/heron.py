import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from torch import nn, optim

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfPPO = TypeVar("SelfPPO", bound="PPO")

class RewardModel(nn.Module):
    def __init__(self, obs_shape=17, lr=1e-3, normalize=False, heirarchy=[0,1], sigma=0.0, multiple_sigmas=False):
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.mu = 0
        self.sigma = sigma
        self.multiple_sigmas = multiple_sigmas
        self.normalize = normalize

        self.heirarchy = heirarchy

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.normalize:
            x = (x - self.mu) / self.sigma
        return x

    def get_loss(self, x, reward_signal):
        sign = [1, 1, 1, 1]
        total_loss = 0
        total = 0
        correct = 0
        outs = []
        for i in range(x.shape[0]):
            
            j = np.random.randint(x.shape[0])
            while i == j:
                j = np.random.randint(x.shape[0])

            x_i = x[i]
            x_j = x[j]

            reward_i = self(x_i)
            reward_j = self(x_j)
            

            reward_info_i = reward_signal[i]
            reward_info_j = reward_signal[j]

            if not self.multiple_sigmas:
                self.sigma = [self.sigma] * reward_info_i.shape[0]

            # Level 1
            if reward_info_i[self.heirarchy[0]] * sign[0] > reward_info_j[self.heirarchy[0]] * sign[0] + self.sigma[0] * reward_signal[:, self.heirarchy[0]].std():
                loss = -1 * th.log(th.sigmoid(reward_i - reward_j))
                if reward_i > reward_j:
                    correct += 1
            elif reward_info_j[self.heirarchy[0]] * sign[0] > reward_info_i[self.heirarchy[0]] * sign[0] + self.sigma[0] * reward_signal[:, self.heirarchy[0]].std():
                loss = -1 * th.log(th.sigmoid(reward_j - reward_i))
                if reward_j > reward_i:
                    correct += 1
            elif reward_info_i.shape[0] == 1:
                continue
            # Level 2
            elif reward_info_i[self.heirarchy[1]] * sign[1] > reward_info_j[self.heirarchy[1]] * sign[1] + self.sigma[1] * reward_signal[:, self.heirarchy[1]].std():
                loss = -1 * th.log(th.sigmoid(reward_i - reward_j))
                if reward_i > reward_j:
                    correct += 1
            elif reward_info_j[self.heirarchy[1]] * sign[1] > reward_info_i[self.heirarchy[1]] * sign[1] + self.sigma[1] * reward_signal[:, self.heirarchy[1]].std():
                loss = -1 * th.log(th.sigmoid(reward_j - reward_i))
                if reward_j > reward_i:
                    correct += 1
            elif reward_info_i.shape[0] == 2:
                continue
            # Level 3
            elif reward_info_i[self.heirarchy[2]] * sign[2] > reward_info_j[self.heirarchy[2]] * sign[2] + self.sigma[1] * reward_signal[:, self.heirarchy[2]].std():
                loss = -1 * th.log(th.sigmoid(reward_i - reward_j))
                if reward_i > reward_j:
                    correct += 1
            elif reward_info_j[self.heirarchy[2]] * sign[2] > reward_info_i[self.heirarchy[2]] * sign[2] + self.sigma[1] * reward_signal[:, self.heirarchy[2]].std():
                loss = -1 * th.log(th.sigmoid(reward_j - reward_i))
                if reward_j > reward_i:
                    correct += 1
            elif reward_info_i.shape[0] == 3:
                continue
            # Level 4
            elif reward_info_i[self.heirarchy[3]] * sign[3] > reward_info_j[self.heirarchy[3]] * sign[3] + self.sigma[1] * reward_signal[:, self.heirarchy[3]].std():
                loss = -1 * th.log(th.sigmoid(reward_i - reward_j))
                if reward_i > reward_j:
                    correct += 1
            elif reward_info_j[self.heirarchy[3]] * sign[3] > reward_info_i[self.heirarchy[3]] * sign[3] + self.sigma[1] * reward_signal[:, self.heirarchy[3]].std():
                loss = -1 * th.log(th.sigmoid(reward_j - reward_i))
                if reward_j > reward_i:
                    correct += 1
            elif reward_info_i.shape[0] == 4:
                continue
            else:
                continue
            total += 1
            total_loss += loss
        return total_loss / (total + 1e-5), correct / (total + 1e-5), outs


class RLHFRewardModel(nn.Module):
    def __init__(self, obs_shape=17, lr=1e-3, normalize=False, heirarchy=[0,1], sigma=0.0, multiple_sigmas=False):
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.mu = 0
        self.sigma = sigma
        self.multiple_sigmas = multiple_sigmas
        self.normalize = normalize

        self.heirarchy = heirarchy

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.normalize:
            x = (x - self.mu) / self.sigma
        return x

    def get_loss(self, x, reward_signal):
        sign = [1, 1, 1, 1]
        total_loss = 0
        total = 0
        correct = 0
        outs = []
        for i in range(x.shape[0]):
            
            j = np.random.randint(x.shape[0])
            while i == j:
                j = np.random.randint(x.shape[0])

            x_i = x[i]
            x_j = x[j]

            reward_i = self(x_i)
            reward_j = self(x_j)
            

            reward_info_i = reward_signal[i].sum()
            reward_info_j = reward_signal[j].sum()

            if reward_info_i > reward_info_j:
                loss = -1 * th.log(th.sigmoid(reward_i - reward_j))
                if reward_i > reward_j:
                    correct += 1
            elif reward_info_j > reward_info_i:
                loss = -1 * th.log(th.sigmoid(reward_j - reward_i))
                if reward_j > reward_i:
                    correct += 1
            
            total += 1
            total_loss += loss
        return total_loss / (total + 1e-5), correct / (total + 1e-5), outs

class HERON(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        heirarchy,
        sigma,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        factor_dim=2,
        multiple_sigmas=False,
        heron=True,
        heuristic=False,
        alpha=0.5,
        rlhf = False
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            factor_dim=factor_dim,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.RM = None
        self.heirarchy=heirarchy
        self.sigma=sigma
        self.multiple_sigmas=multiple_sigmas
        self.heron=heron
        self.heuristic=heuristic
        self.alpha=alpha
        self.rlhf=rlhf

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # Do a complete pass on the rollout buffer to train the RM
        total_reward = []
        total_acc = []
        factor_min = None
        facot_max = None
        if self.heron:
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                if self.RM is None:
                    print(self.rlhf)
                    if self.rlhf:
                        self.RM = RLHFRewardModel(obs_shape = rollout_data.observations.shape[1], lr=1e-3, heirarchy=self.heirarchy, sigma=self.sigma, multiple_sigmas=self.multiple_sigmas)
                    else:
                        self.RM = RewardModel(obs_shape = rollout_data.observations.shape[1], lr=1e-3, heirarchy=self.heirarchy, sigma=self.sigma, multiple_sigmas=self.multiple_sigmas)

                factors = rollout_data.factors
                obs = rollout_data.observations
                
                self.RM.optimizer.zero_grad()
                loss, acc, _ = self.RM.get_loss(x = obs, reward_signal=factors)
                loss.backward()

                self.RM.optimizer.step()
                total_reward += [rollout_data.returns.mean()]
                total_acc += [acc]

            print(np.array(total_reward).mean())
            print("ACC", np.array(total_acc).mean())

        if self.heuristic:
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                factors = rollout_data.factors
                if factor_min is None:
                    factor_min = factors.min(0)[0]
                    factor_max = factors.max(0)[0]
                    print(factors.min(0))

                for k in range(factors.shape[1]):
                    factor_min[k] = min(factor_min[k], factors.min(0)[0][k])
                    factor_max[k] = min(factor_max[k], factors.max(0)[0][k])

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):

            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                # print(rollout_data)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target

                if self.heron:
                    pred_returns = self.RM(rollout_data.observations).squeeze()
                    value_loss = F.mse_loss(pred_returns, values_pred)
                elif self.heuristic:

                    reward = th.zeros([rollout_data.factors.shape[0]])
                    if rollout_data.factors.shape[1] == 4:
                        reward = rollout_data.factors[:, 1]
                    else:
                        reward = rollout_data.factors[:, 0]
                    # for p in range(1, rollout_data.factors.shape[1] + 1):
                    #     # f = rollout_data.factors[:, self.heirarchy[p-1]]
                    #     # f = (f - factor_min[self.heirarchy[p-1]]) / (factor_max[self.heirarchy[p-1]] - factor_min[self.heirarchy[p-1]] + 1e-6)
                    #     reward += f * self.alpha ** p
                    value_loss = F.mse_loss(reward, values_pred)
                else:
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
