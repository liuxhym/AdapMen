from __future__ import annotations

import gym
from gym.wrappers import TimeLimit
import numpy as np


class AbsorbingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(AbsorbingWrapper, self).__init__(env)
        obs_space = self.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            shape=(obs_space.shape[0] + 1,),
            low=obs_space.low[0],
            high=obs_space.high[0],
            dtype=obs_space.dtype)

    def observation(self, observation):
        return self.get_non_absorbing_state(observation)

    @staticmethod
    def get_non_absorbing_state(obs):
        return np.concatenate([obs, [0]], -1)

    def get_absorbing_state(self):
        obs = np.zeros(self.observation_space.shape)
        obs[-1] = 1
        return obs

    @property
    def _max_episode_steps(self):
        try: 
            self.env: TimeLimit
            return self.env._max_episode_steps  # pylint: disable=protected-access
        except:
            return 500
