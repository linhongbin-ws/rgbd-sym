from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np
from copy import deepcopy as cp
import gym
import cv2


class GymRegularizer(BaseWrapper):
    def __init__(self, env,
                 **kwargs,
                 ):
        super().__init__(env,)

    def reset(self,):
        obs = self.env.reset()
        obs = self._obs_proc(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_proc(obs)
        return obs, reward, done, info
    
    def _obs_proc(self , obs):
        new_obs = cp(obs['image'])

        s = new_obs.shape[1:]
        s2 = obs['occup_image'].shape
        if s[0]!= s2[0] or s[1]!= s2[1]:
            occup_image = cv2.resize(obs['occup_image'],
                        s,
                        interpolation=cv2.INTER_AREA)
        else:
            occup_image = obs['occup_image']
        new_obs[0,:,:] = occup_image
        return new_obs
    # @property
    # def observation_space(self):
    #     obs = self.env.observation_space
    #     new_obs = {k: v for k,v in obs.items() if k in self._obs_key}
    #     return gym.spaces.Dict(new_obs)