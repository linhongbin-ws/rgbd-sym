from rgbd_sym.env.embodied.base import BaseEnv
import gym
from rgbd_sym.env.embodied import pomdp
import numpy as np
from copy import deepcopy as cp
import pybullet as pd
from rgbd_sym.tool.common import getT

class DummyEnv(BaseEnv):
    def __init__(self,
                 task,
                 delta_transl= 1,
                 delta_rot=  np.pi / 8,
                 **args):
        self._task = task
        client = None
        self._seed = 0
        super().__init__(client)

        self._delta_transl = delta_transl
        self._delta_rot = delta_rot
        

    def get_oracle_action(self):
        pass

    def render(self):
        imgs ={}
        imgs['pc'] = cp(self._points)
        return imgs
    
    def reset(self):
        obs = {}
        obs['pc'] = cp(self._points)
        return obs
    
    def step(self, action):
        _transform_dict = {}
        _transform_dict['gripper'] = getT([0, 0, 0], 
                                          [0, 0, -action[4]*self._delta_rot], 
                                          rot_type="euler", euler_Degrees=False)
        _transform_dict['object1'] = getT([-self._delta_transl*action[2],
                                        -self._delta_transl*action[1],
                                        self._delta_transl*action[3],],
                                        [0, 0, 0],
                                        rot_type="euler")
        _transform_dict['object2'] = cp(_transform_dict['object1'])
        _transform_dict['object3'] = cp(_transform_dict['object1'])
        
        new_points = {}
        for k, _pc in self._points.items():
            ones = np.ones((_pc.shape[0], 1))
            P = np.concatenate((_pc, ones), axis=1)
            new_points[k] = np.matmul(P, np.transpose(_transform_dict[k]))[:, :3]
        self._points = new_points

        
        obs = {}
        obs['pc'] = cp(self._points)
        done = False
        reward = 0 
        info = {}
        return obs, reward, done,  info

    def set_current_points(self, points):
        self._points = cp(points)

    
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.client.seed(self._seed)

    @property
    def image_space(self):
        return self.client.image_space
    
    @property
    def observation_space(self):
        return self.client.observation_space

    @property
    def action_space(self):
        return self.client.action_space

