from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np
from rgbd_sym.tool.sym import local_sym_step, get_sym_params
from rgbd_sym.tool.common import scale_arr
import cv2
import gym

class SymObs(BaseWrapper):
    def __init__(self, env, depth_offset=5, sym_image_size=84, skip=False, **kwargs):
        super().__init__(env, **kwargs)
        self._sym_args = get_sym_params(env_name=self.unwrapped.task)
        self._sym_args['K'] = self.unwrapped.instrinsic_K
        self._depth_offset = depth_offset
        self._sym_image_size  = sym_image_size
        self._new_obs_shape = None
        self._skip = skip

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['image'] = self._proc_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs['image']  = self._proc_obs(obs)
        return obs

    def _proc_obs(self, obs):
        if self._skip:
            masks = obs['mask']
            gripper_min =  np.min(obs['depth']["gripper"][obs['mask']["gripper"]])
            new_d = {}
            for k, _ in obs['mask'].items():
                _d = obs['depth'][k].copy()
                _d[obs['mask'][k]] = _d[obs['mask'][k]] - gripper_min
                new_d[k] = _d

            depth_img = None
            for k,v in new_d.items():
                depth_img = v if depth_img is None else np.minimum(depth_img, v)
        else:

            depth_img, masks = local_sym_step(obs['depth'],
                                    obs['mask'], [],
                                    **self.sym_args)
            depth_img = depth_img[0]
            masks = masks[0]
        background_mask = None
        for k, m in masks.items():
            background_mask = m if background_mask is None else np.logical_or(background_mask, m)
        background_depth = np.max(depth_img[background_mask])
        depth_img[np.logical_not(background_mask)] = background_depth + self._depth_offset
        depth_img = np.clip(depth_img,0, 255)
        depth_real  =depth_img / 255

        if self._skip:
            l = 150
            depth_real = depth_real[300-l:300+l,300-l:300+l]
        depth_real = cv2.resize(depth_real, (self._sym_image_size, self._sym_image_size), interpolation=cv2.INTER_NEAREST)
        scalar_layer = np.ones(depth_real.shape, dtype=np.uint8) * obs['grasp_sig']
        new_img = np.stack([depth_real, scalar_layer], axis=0)

        return new_img
    @property
    def sym_args(self):
        return self._sym_args
    @property
    def observation_space(self):
        if self._new_obs_shape is None:
            obs = self.reset()
            self._new_obs_shape = obs['image'].shape
        return gym.spaces.Box(0, 1, self._new_obs_shape,
                              dtype=np.float64)
