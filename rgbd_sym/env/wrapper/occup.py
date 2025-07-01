from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np
from copy import deepcopy as cp
# from rgbd_sym.tool.o3d import pointclouds2occupancy
from rgbd_sym.tool.depth import pointclouds2occupancy
from rgbd_sym.tool.depth import occup2image
class Occup(BaseWrapper):
    def __init__(self, env, 
                 out_image_type = 'depth',
                 out_background_encoding = 255,
                 pc_x_min = -0.2,
                 pc_y_min = -0.2,
                 pc_z_min = 0,
                 pc_range = 0.4,
                 occup_res = 200,
                 **kwargs):
        super().__init__(env, **kwargs)
        self._out_background_encoding = out_background_encoding
        self._out_image_type = out_image_type
        self._pc_range = pc_range
        self._pc_x_min = pc_x_min
        self._pc_y_min = pc_y_min
        self._pc_z_min = pc_z_min
        self._occup_res = occup_res

        # if self.unwrapped._task == "block_push":
        #     # self._pc_range =  0.2
        #     # self._pc_x_min = -0.1
        #     # self._pc_y_min = -0.1
        #     # self._pc_z_min = 0
        #     self._occup_res = 200
        # if self.unwrapped._task == "block_pick":
        #     # self._pc_range =  0.2
        #     # self._pc_x_min = -0.1
        #     # self._pc_y_min = -0.1
        #     # self._pc_z_min = 0
        #     self._occup_res = 200




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

        pc_offset = np.min(points[:,2])
        points[:,2] = points[:,2] - pc_offset
        # print(f"x: {np.min(points[:,0])} {np.max(points[:,0])}", end= " ")
        # print(f"y: {np.min(points[:,1])} {np.max(points[:,1])}", end= " ")
        # print(f"z: {np.min(points[:,2])} {np.max(points[:,2])}",)
        occ_mat = pointclouds2occupancy(
            points,
            occup_h=self._occup_res,
            occup_w=self._occup_res,
            occup_d=self._occup_res,
            pc_x_min=self._pc_x_min,
            pc_x_max= self._pc_x_min + self._pc_range, 
            pc_y_min= self._pc_y_min, 
            pc_y_max=self._pc_y_min + self._pc_range, 
            pc_z_min= 0, 
            pc_z_max= self._pc_range,
        )
        z, z_mask = occup2image(occ_mat, 
                                image_type="real_depth",
                                depth_min = 0,
                                depth_max = self._pc_range)
        # import matplotlib.pyplot as plt
        # from matplotlib.pyplot import imshow, subplot, axis, cm, show
        # imshow(z_mask)
        # plt.colorbar()
        # plt.show()
        # z = -z
        if self.unwrapped._task =="block_push":
            points = cp(obs['pc']['goal'])
            points[:,2] = points[:,2] - pc_offset

            occ_mat = pointclouds2occupancy(
                points,
                occup_h=self._occup_res,
                occup_w=self._occup_res,
                occup_d=self._occup_res,
                pc_x_min=self._pc_x_min,
                pc_x_max= self._pc_x_min + self._pc_range, 
                pc_y_min= self._pc_y_min, 
                pc_y_max=self._pc_y_min + self._pc_range, 
                pc_z_min= 0, 
                pc_z_max= self._pc_range,
            )
            _, goal_mask = occup2image(occ_mat, 
                                    image_type="real_depth",
                                    depth_min = 0,
                                    depth_max = self._pc_range)
            if np.any(goal_mask):
                z[np.logical_not(z_mask)] = np.max(z[z_mask]) - 0.02 # the offset is the background depth
            else:
                z[np.logical_not(z_mask)] = np.max(z[z_mask]) + 0.07 # the offset is the background depth
        else:
            z[np.logical_not(z_mask)] = np.max(z[z_mask]) + 0.07 # the offset is the background depth
            
        z = np.transpose(z)
        # z = np.flip(z, axis=0)
    
        new_obs['occup_image'] = z
        return new_obs