import pytest
import numpy as np

from gym.spaces import Box, Tuple, Discrete, MultiDiscrete
from tests.vector.utils import CustomSpace, make_env, make_custom_space_env

from gym.vector.sync_vector_env import SyncVectorEnv


def test_create_sync_vector_env():
    env_fns = [make_env("FrozenLake-v1", i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
    finally:
        env.close()

    assert env.num_envs == 8


def test_reset_sync_vector_env():
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
        observations = env.reset()
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    del observations

    try:
        env = SyncVectorEnv(env_fns)
        observations = env.reset(return_info=False)
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    del observations

    env_fns = [make_env("CartPole-v1", i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
        observations, infos = env.reset(return_info=True)
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape
    assert isinstance(infos, list)
    assert all([isinstance(info, dict) for info in infos])


@pytest.mark.parametrize("use_single_action_space", [True, False])
def test_step_sync_vector_env(use_single_action_space):
    env_fns = [make_env("FrozenLake-v1", i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
        observations = env.reset()

        assert isinstance(env.single_action_space, Discrete)
        assert isinstance(env.action_space, MultiDiscrete)

        if use_single_action_space:
            actions = [env.single_action_space.sample() for _ in range(8)]
        else:
            actions = env.action_space.sample()
        observations, rewards, dones, _ = env.step(actions)
    finally:
        env.close()

    assert isinstance(env.observation_space, MultiDiscrete)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    assert isinstance(rewards, np.ndarray)
    assert isinstance(rewards[0], (float, np.floating))
    assert rewards.ndim == 1
    assert rewards.size == 8

    assert isinstance(dones, np.ndarray)
    assert dones.dtype == np.bool_
    assert dones.ndim == 1
    assert dones.size == 8


def test_call_sync_vector_env():
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]
    try:
        env = SyncVectorEnv(env_fns)
        _ = env.reset()
        images = env.call("render", mode="rgb_array")
        gravity = env.call("gravity")
    finally:
        env.close()

    assert isinstance(images, tuple)
    assert len(images) == 4
    for i in range(4):
        assert isinstance(images[i], np.ndarray)

    assert isinstance(gravity, tuple)
    assert len(gravity) == 4
    for i in range(4):
        assert isinstance(gravity[i], float)
        assert gravity[i] == 9.8


def test_set_attr_sync_vector_env():
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]
    try:
        env = SyncVectorEnv(env_fns)
        env.set_attr("gravity", [9.81, 3.72, 8.87, 1.62])
        gravity = env.get_attr("gravity")
        assert gravity == (9.81, 3.72, 8.87, 1.62)
    finally:
        env.close()


def test_check_spaces_sync_vector_env():
    # CartPole-v1 - observation_space: Box(4,), action_space: Discrete(2)
    env_fns = [make_env("CartPole-v1", i) for i in range(8)]
    # FrozenLake-v1 - Discrete(16), action_space: Discrete(4)
    env_fns[1] = make_env("FrozenLake-v1", 1)
    with pytest.raises(RuntimeError):
        env = SyncVectorEnv(env_fns)
        env.close()


def test_custom_space_sync_vector_env():
    env_fns = [make_custom_space_env(i) for i in range(4)]
    try:
        env = SyncVectorEnv(env_fns)
        reset_observations = env.reset()

        assert isinstance(env.single_action_space, CustomSpace)
        assert isinstance(env.action_space, Tuple)

        actions = ("action-2", "action-3", "action-5", "action-7")
        step_observations, rewards, dones, _ = env.step(actions)
    finally:
        env.close()

    assert isinstance(env.single_observation_space, CustomSpace)
    assert isinstance(env.observation_space, Tuple)

    assert isinstance(reset_observations, tuple)
    assert reset_observations == ("reset", "reset", "reset", "reset")

    assert isinstance(step_observations, tuple)
    assert step_observations == (
        "step(action-2)",
        "step(action-3)",
        "step(action-5)",
        "step(action-7)",
    )
