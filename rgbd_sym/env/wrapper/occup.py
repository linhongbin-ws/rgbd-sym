from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np
from copy import deepcopy as cp
from rgbd_sym.tool.o3d import pointclouds2occupancy
from rgbd_sym.tool.depth import occup2image
class Occup(BaseWrapper):
    def __init__(self, env, 
                 out_image_type = 'depth',
                 out_background_encoding = 255,
                 pc_x_min = -0.2,
                 pc_y_min = -0.2,
                 pc_z_min = -1,
                 pc_range = 0.4,
                 occup_res = 84,
                 **kwargs):
        super().__init__(env, **kwargs)
        self._out_background_encoding = out_background_encoding
        self._out_image_type = out_image_type
        self._pc_range = pc_range
        self._pc_x_min = pc_x_min
        self._pc_y_min = pc_y_min
        self._pc_z_min = pc_z_min
        self._occup_res = occup_res

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._proc_obs(obs)
        return obs, reward, done, info
    def reset(self):
        obs = self.env.reset()
        obs = self._proc_obs(obs)
        return obs

    def _proc_obs(self, obs):
        new_obs = cp(obs)
        points = None
        for k,v in obs['pc'].items():
            points = v if points is None else np.concatenate((points, v), axis=0)
        occ_mat = pointclouds2occupancy(
            points,
            occup_h=self._occup_res,
            occup_w=self._occup_res,
            occup_d=self._occup_res,
            pc_x_min=self._pc_x_min,
            pc_x_max= self._pc_x_min + self._pc_range, 
            pc_y_min= self._pc_y_min, 
            pc_y_max=self._pc_y_min + self._pc_range, 
            pc_z_min= self._pc_z_min, 
            pc_z_max=self._pc_z_min + self._pc_range,
        )
        z, z_mask = occup2image(occ_mat, 
                                image_type=self._out_image_type,
                                background_encoding=self._out_background_encoding)
        new_obs['occup_image'] = z
        return new_obs