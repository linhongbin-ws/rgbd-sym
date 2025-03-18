from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np
from rgbd_sym.tool.sym import local_sym_step, get_sym_params


class SymObs(BaseWrapper):
    def __init__(self, env, depth_offset=15, **kwargs):
        super().__init__(env, **kwargs)
        self._sym_args = get_sym_params(env_name=self.unwrapped.task)
        self._sym_args['K'] = self.unwrapped.instrinsic_K
        self._depth_offset = depth_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs['image'] = self._proc_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs['image'] = self._proc_obs(obs)
        return obs

    def _proc_obs(self, obs):

        depth_img, masks = local_sym_step(obs['depth'],
                                   obs['mask'], [],
                                   **self.sym_args)
        depth_img = depth_img[0]
        masks = masks[0]
        maxs = []
        background_mask = None
        for k, m in masks.items():
            # maxs.append(np.max(depth_img[m]))
            background_mask = m if background_mask is None else np.logical_or(background_mask, m)
        background_depth = np.max(depth_img[background_mask])
        depth_img[np.logical_not(background_mask)] = background_depth + self._depth_offset
        depth_img = np.clip(depth_img,0, 255)
        depth_real  =depth_img / 255
        new_img = np.stack([depth_real, obs['image'][1, :, :]], axis=0)
        return new_img
    @property
    def sym_args(self):
        return self._sym_args
