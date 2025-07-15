from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np
from copy import deepcopy as cp
import gym
import cv2


class GymRegularizer(BaseWrapper):
    def __init__(self, env,
                 obs_type = "occup",
                 **kwargs,
                 ):
        self._obs_type = obs_type
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
        if self._obs_type == "occup":
            layer0 = cp(obs['occup_image'])
        elif self._obs_type == "image":
             layer0 = cp(obs['image'][0,:,:])

        layer1 = cp(obs['image'][1,:,:])

        s = 84
        layer0 = cv2.resize(layer0, (s,s), interpolation=cv2.INTER_AREA)
        # layer0 = cv2.resize(layer0, (s,s), interpolation=cv2.INTER_NEAREST)
        layer1 = np.ones((s,s), dtype=layer1.dtype) * layer1[0][0]
        new_obs = np.stack((layer0, layer1), axis=0)

        # s = new_obs.shape[1:]
        # s2 = obs['occup_image'].shape
        # if s[0]!= s2[0] or s[1]!= s2[1]:
        #     occup_image = cv2.resize(obs['occup_image'],
        #                 s,
        #                 interpolation=cv2.INTER_AREA)
        # else:
        #     occup_image = obs['occup_image']
        # new_obs[0,:,:] = occup_image
        return new_obs
    
    @property
    def observation_space(self):
        obs_space = gym.spaces.Box(low=0, high=1,shape=(2,84,84), dtype=np.float32)
        return obs_space