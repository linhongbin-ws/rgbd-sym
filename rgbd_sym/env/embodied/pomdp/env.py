# from pdomains import *
from rgbd_sym.env.embodied.base import BaseEnv
import gym
from rgbd_sym.env.embodied import pomdp
import numpy as np
from copy import deepcopy as cp
import pybullet as pd
from rgbd_sym.tool.depth import projection_matrix_to_K
from rgbd_sym.tool.o3d import depth_image_to_point_cloud
# from pomdp_envs import pomdp
# from rgbd_sym.tool.common import scale_arr
# from copy import deepcopy
# from rgbd_sym.tool.depth import projection_matrix_to_K


class PomdpEnv(BaseEnv):
    def __init__(self, 
                 task='block_pull',
                 ):
        if task == 'block_pull':
            client = gym.make("BlockPull-Sym")
        elif task == 'block_pick':
            client = gym.make("BlockPick-Sym")
        elif task == 'block_push':
            client = gym.make("BlockPush-Sym")
        elif task == 'drawer_open':
            client = gym.make("DrawerOpen-Sym")
        else:
            raise NotImplementedError
    
        super().__init__(client)
        self._task = task
        self._obs = None

    def reset(self):
        self._obs = self.client.reset()
        self._obs = self._obs_proc(self._obs)
        return self._obs
    def step(self,action):
        self._obs, reward, done, info = self.client.step(action)
        self._obs = self._obs_proc(self._obs)
        # print(self._obs['gripper_pos'])
        return self._obs, reward, done, info
    def render(self):   
        return self._obs
    
    def query_expert(self, eps):
        return self.client.query_expert(eps)

    def _mask_or(self, mask, ids):
        x = None
        for _id in ids:
            out = mask == _id
            x = out if x is None else x | out
        return x

    
    def _obs_proc(self, obs):
        new_obs = cp(obs)
        # mask process
        mask_metadata = cp(obs['mask_metadata'])
        get_mask = lambda in_obj_data, in_link_data,  _obj_id, _obj_link_id: (in_obj_data == _obj_id) & (self._mask_or(in_link_data, _obj_link_id))
        masks = {}
        masks['gripper'] =  cp(obs['gripper_mask'])



        
        if self._task in ['block_pull','block_pick', 'block_push']:
            _obj_ids = [o.object_id for o in self.client.core_env.objects]
            for i, o_id in enumerate(_obj_ids):
                masks['object'+str(i+1)] = get_mask(mask_metadata[0], mask_metadata[1], o_id, [-1])
        elif self._task in ['drawer_open']:
            _obj_ids = [self.client.core_env.drawer,self.client.core_env.locked_drawer,]
            for i, o_id in enumerate(_obj_ids):
                # links_ids = [2,3,4,6,7]
                # links_ids = [11]
                links_ids = np.arange(12).tolist()
                # print("xxxxxxxxxx")
                # print(links_ids)
                masks['object'+str(i+1)] = get_mask(mask_metadata[0], mask_metadata[1], o_id.id,links_ids)     #2,3,4 6 7
                handle = get_mask(mask_metadata[0], mask_metadata[1], o_id.handle.id, [-1,0,1])
                masks['object'+str(i+1)] = np.logical_or(masks['object'+str(i+1)], handle)
        


        
        # else:
        #     _obj_ids = [self.drawer,self.locked_drawer,]
        #     for i, o_id in enumerate(_obj_ids):
        #     # links_ids = [2,3,4,6,7]
        #     # links_ids = [11]
        #     links_ids = np.arange(12).tolist()
        #     # print("xxxxxxxxxx")
        #     # print(links_ids)
        #     masks['object'+str(i+1)] = np.transpose(get_mask(mask_metadata[0], mask_metadata[1], o_id.id,links_ids))      #2,3,4 6 7
        #     handle = np.transpose(get_mask(mask_metadata[0], mask_metadata[1], o_id.handle.id, [-1,0,1]))
        #     masks['object'+str(i+1)] = np.logical_or(masks['object'+str(i+1)], handle)
        #     return new_obs
        
        # https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        if self._task in ['block_push']:
            goal_mask =  cp(obs['goal_mask']) # make sure "goal" at the last key

            for k, v in masks.items():
                goal_mask = goal_mask & np.logical_not(v)
            masks['goal'] = goal_mask

        pc_dict = {}
        for k, v in masks.items():
            depthImg = cp(obs['depth'][0])
            if True:
                if k not in ["gripper","goal"]: # if objects are occluded by gripper, then fill missing pixels
                    mask_except_gripper = v & np.logical_not(masks['gripper'])
                    obj_d = np.mean(depthImg[mask_except_gripper])
                    mask_overlap_gripper = v & masks['gripper']
                    depthImg[mask_overlap_gripper]  = obj_d

                depth_real = depthImg
                encode_mask = np.zeros(depth_real.shape, dtype=np.uint8)
                encode_mask[v] = 1
                proj = np.array(list(obs['proj_mat'])).reshape(4,4)
                new_K = projection_matrix_to_K(proj, depth_real.shape[0])
                scale = 1
                pose = np.eye(4)
                rgb = np.zeros(depth_real.shape + (3,), dtype=np.uint8)
                points = depth_image_to_point_cloud(
                    rgb, depth_real, scale, new_K, pose, encode_mask=encode_mask, tolist=False
                )
                points = points[points[:, 6] == 1, :3]  # remove background
            else:
                ## note: this part of code do not work well! Point cloud Z do not sync with the depth

                # import matplotlib.pyplot as plt
                # from matplotlib.pyplot import imshow, subplot, axis, cm, show
                # imshow(depthImg)
                # plt.colorbar()
                # plt.show()
                depthImg[np.logical_not(v)] = 10000
                # imshow(depthImg)
                # plt.colorbar()
                # plt.show()
                size = depthImg.shape[0]
                projectionMatrix = np.asarray(obs['proj_mat']).reshape([4,4],order='F')
                viewMatrix = np.asarray(obs['view_mat']).reshape([4,4],order='F')
                tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
                pixel_pos = np.mgrid[0:size, 0:size]
                pixel_pos = pixel_pos/(size/2) - 1
                pixel_pos = np.moveaxis(pixel_pos, 1, 2)
                pixel_pos[1] = -pixel_pos[1]
                zs = 2*depthImg.reshape(1, size, size) - 1
                pixel_pos = np.concatenate((pixel_pos, zs))
                pixel_pos = pixel_pos.reshape(3, -1)
                augment = np.ones((1, pixel_pos.shape[1]))
                pixel_pos = np.concatenate((pixel_pos, augment), axis=0)
                position = np.matmul(tran_pix_world, pixel_pos)
                pc = position / position[3]
                points = pc.T[:, :3]
                
                points = np.array(points)
                threshold = -10
                points = points[points[:,2]>threshold,:]

                ref = obs['gripper_pos']
                ref[2] = 1
                points = points - ref
                # points[:,2] = points[:,2] * 3  # scale depth 
                # if k!="gripper":
                #     points[:,2] = points[:,2] - 0.3 # scale depth 
                # if k == "gripper":
                #     print(f"x: {np.min(points[:,0])} {np.max(points[:,0])}", end= " ")
                #     print(f"y: {np.min(points[:,1])} {np.max(points[:,1])}", end= " ")
                #     print(f"z: {np.min(points[:,2])} {np.max(points[:,2])}",)
                # if points.shape[0]>0:
                #     print("x range:", np.min(points[:,0]),np.max(points[:,0]))
                #     print("y range:", np.min(points[:,1]),np.max(points[:,1]))
                #     print("z range:", np.min(points[:,2]),np.max(points[:,2]))

                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(points)
                # o3d.visualization.draw_geometries([pcd])
            
            pc_dict[k] = points
        
        # print(f"gripper to object max: {np.max(obs['depth'][0][masks['gripper']])- np.max(obs['depth'][0][masks['object1']])}")
        # print(f"gripper to object max: {np.max(pc_dict['gripper'][:,2])-np.max(pc_dict['object1'][:,2])}")
        # print(f"gripper to object min: {np.min(pc_dict['gripper'][:,2])-np.min(pc_dict['object1'][:,2])}")

        
        new_obs['mask'] = masks
        new_obs['pc'] = pc_dict
        return new_obs
    

    
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
    
    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        # if name[0] == "_":
        #     raise Exception("cannot find {}".format(name))
        # else:
        return getattr(self.client, name)