from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np
from copy import deepcopy
import gym


class GymRegularizer(BaseWrapper):
    def __init__(self, env,
                 obs_key=['image','vector'],
                 **kwargs,
                 ):
        super().__init__(env,)
        self._obs_key = obs_key

    def reset(self,):
        obs = self.env.reset()
        obs = self._obs_proc(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_proc(obs)
        return obs, reward, done, info
    
    def _obs_proc(self , obs):
        new_obs = {}

        if "all" in self._obs_key:
            new_obs = obs.copy()
        else:
            for v in self._obs_key:
                new_obs[v] = obs[v]
        return new_obs

    @property
    def observation_space(self):
        obs = self.env.observation_space
        new_obs = {k: v for k,v in obs.items() if k in self._obs_key}
        return gym.spaces.Dict(new_obs)