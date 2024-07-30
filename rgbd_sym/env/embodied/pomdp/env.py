from pdomains import *
from rgbd_sym.env.embodied.base import BaseEnv
import gym
import numpy as np

class PomdpEnv(BaseEnv):
    def __init__(self,
                 task,
                pybullet_gui=False,
                  **kwargs,):
        if task== 'block_picking':
            task_id = 'pdomains-block-picking-pixel-v0'
        client=gym.make(task_id, rendering=pybullet_gui)
        super().__init__(client)
        obs = self.client.reset()
        obs = self._process_obs(obs)
        self._new_obs_shape = {k: v.shape for k, v in obs.items()}

    def reset(self):
        self.timestep = 0
        obs = self.client.reset()
        obs = self._process_obs(obs)
        obs['is_success'] = 0
        return obs
    
    def step(self, action):
        self.timestep += 1
        obs, reward, done, info = self.client.step(action)
        obs = self._process_obs(obs)
        obs['is_success'] = 1 if info['success'] else 0
        return obs, reward, done, info

    def render(self, mode="human"):  # ['human', 'rgb_array', 'mask_array']
        return self.client.render(mode=mode)

    def get_oracle_action(self, obs=None):
        return self.client.query_expert(0)
    
    def _process_obs(self, _obs):
        obs = _obs.copy()
        obs_t = np.transpose(obs, axes=[2,1,0])
        obs_t = np.concatenate([obs_t, np.zeros(obs_t.shape[:2]+(1,), dtype=np.uint8)],axis=2)
        new_obs = {}
        new_obs['image'] = obs_t
        return new_obs
    @property
    def observation_space(self):
        obs = {}
        obs['image'] = gym.spaces.Box(0, 255, self._new_obs_shape["image"],
                                          dtype=np.uint8)
        obs['is_success'] = gym.spaces.Discrete(2)

        
        return gym.spaces.Dict(obs)



if __name__ == "__main__":
    env = POMDP_ENV(task_name='pdomains-block-picking-v0')
    obs = env.reset()
    done = False

    while not done:
        # action = env.action_space.sample()
        action = env.get_oracle_action(obs)
        obs, reward, done, info = env.step(action)
        print(obs)
    print(env.observation_space)
    print(env.action_space)