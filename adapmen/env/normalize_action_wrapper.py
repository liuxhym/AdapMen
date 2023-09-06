from __future__ import annotations

import gym
from gym import spaces
from gym.wrappers import TimeLimit
import numpy as np


class NormalizeBoxActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(NormalizeBoxActionWrapper, self).__init__(env)
        env: TimeLimit
        action_space = env.action_space
        assert isinstance(action_space, spaces.Box)
        self.low, self.high = action_space.low, action_space.high
        self._max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access

    def action(self, action):
        # rescale the action
        scaled_action = self.low + (action + 1.0) * (self.high - self.low) / 2.0
        scaled_action = np.clip(scaled_action, self.low, self.high)

        return scaled_action

    def reverse_action(self, scaled_action):
        action = (scaled_action - self.low) * 2.0 / (self.high - self.low) - 1.0
        return action


def check_and_normalize_box_actions(env):
    low, high = env.action_space.low, env.action_space.high

    if isinstance(env.action_space, spaces.Box):
        if (np.abs(low + np.ones_like(low)).max() > 1e-6 or
                np.abs(high - np.ones_like(high)).max() > 1e-6):
            return NormalizeBoxActionWrapper(env)

    # Environment does not need to be normalized.
    return env
