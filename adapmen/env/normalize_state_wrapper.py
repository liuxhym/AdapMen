from __future__ import annotations

import gym
from gym.wrappers import TimeLimit
import numpy as np


class NormalizeStateWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, shift: np.ndarray, scale: np.ndarray):
        super(NormalizeStateWrapper, self).__init__(env)
        self.shift = shift
        self.scale = scale

    def observation(self, observation: np.ndarray):
        return (observation + self.shift) * self.scale

    @property
    def _max_episode_steps(self):
        self.env: TimeLimit
        return self.env._max_episode_steps  # pylint: disable=protected-access
