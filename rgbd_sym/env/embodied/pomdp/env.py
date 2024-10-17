from pdomains import *
from rgbd_sym.env.embodied.base import BaseEnv
import gym
import numpy as np
from pomdp_envs import pomdp


class PomdpEnv(BaseEnv):
    """ action: [gripper, x,y,z,yaw]"""
    def __init__(self,
                 task,
                pybullet_gui=False,
                  **kwargs,):
        if task== 'block_picking':
            task_id = "BlockPicking-Symm-v0"
        client=gym.make(task_id, rendering=pybullet_gui)
        super().__init__(client)
        obs = self.client.reset()
        obs = self._process_obs(obs)
        # print(obs.keys())
        self._new_obs_shape = {k: v.shape for k, v in obs.items() if k not in ["mask"]}

    def reset(self):
        self.timestep = 0
        obs = self.client.reset()
        obs = self._process_obs(obs)
        # obs['is_success'] = 0
        self._prv_obs = obs
        return obs

    def step(self, action, skip=False):
        _action = action.copy()
        self.timestep += 1
        if skip:
            obs = self._prv_obs
            reward = 0
            done = False
            info = {}
            info["success"] = False
        else:
            obs, reward, done, info = self.client.step(_action)
            obs = self._process_obs(obs)
            # obs['is_success'] = 1 if info['success'] else 0
        return obs, reward, done, info

    def render(self, mode="human"):  # ['human', 'rgb_array', 'mask_array']
        return self.client.render(mode=mode)

    def get_oracle_action(self, obs=None):
        return self.client.query_expert(0)

    def _process_obs(self, _obs):
        new_obs = _obs.copy()
        obs_t = np.transpose(_obs['image'], axes=[2,1,0])
        obs_t = np.concatenate([obs_t, np.zeros(obs_t.shape[:2]+(1,), dtype=np.uint8)],axis=2)
        new_obs['image'] = np.uint8(obs_t*255) # real depth to depth image
        new_obs['depthReal'] =np.transpose(_obs['depth'][0,:,:], axes=[1,0])
        _m = None
        for k,v in new_obs['mask'].items():
            _m = np.logical_or(_m, v) if _m is not None else v
        new_obs['depthReal'][np.logical_not(_m)] = 1 # backgound set to 1 meter
        new_obs['depth'] =np.uint8(new_obs['depthReal']*255)
        return new_obs
    
    @property
    def observation_space(self):
        obs = {}
        obs['image'] = gym.spaces.Box(0, 255, self._new_obs_shape["image"],
                                          dtype=np.uint8)
        # obs['is_success'] = gym.spaces.Discrete(2)

        return gym.spaces.Dict(obs)
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.client.seed(seed)
        self.client.core_env.pose_rng(seed)



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
