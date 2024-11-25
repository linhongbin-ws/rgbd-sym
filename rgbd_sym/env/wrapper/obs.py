from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np


class Obs(BaseWrapper):
    def __init__(self, env, rl_type, 
                 **kwargs):
        super().__init__(env, **kwargs)
        self._rl_type = rl_type

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
    def reset(self):
        obs = self.env.reset()

    def _proc(self, obs):
        if self._rl_type == "rsac":
            info_meta = obs.copy()
            _obs = np.transpose(_obs["image"], axes=[2,0,1])
            _obs = _obs[:2,:,:]
        return _obs, info_meta
    @property
    def observation_space(self):
        if self._new_obs_shape is None:
            obs = self.reset()
            self._new_obs_shape = {k: v.shape for k, v in obs.items() if k not in ["mask","depth"]}
        obs = {}
        obs['image'] = gym.spaces.Box(0, 255, self._new_obs_shape["image"],
                                          dtype=np.uint8)
        # obs['is_success'] = gym.spaces.Discrete(2)
        if not self._obs_dict:
            obs = obs['image']
            return obs

        return gym.spaces.Dict(obs)
