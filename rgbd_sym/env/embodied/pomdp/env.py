from pdomains import *
from rgbd_sym.env.embodied.base import BaseEnv
import gym
import numpy as np
from pomdp_envs import pomdp
from rgbd_sym.tool.common import scale_arr
from copy import deepcopy
from rgbd_sym.tool.depth import projection_matrix_to_K


class PomdpEnv(BaseEnv):
    """ action: [gripper, x,y,z,yaw]"""

    def __init__(self,
                 task,
                 pybullet_gui=False,
                 **kwargs,):
        self.task = task
        task_id = {
                "block_pick":"BlockPicking-Symm-Dict",
                "block_pull":"BlockPulling-Symm-Dict",
                "block_push":"BlockPushing-Symm-Dict",
                "drawer_open":"DrawerOpening-Symm-Dict", 
                   }[task]
        # if task == 'block_pick':
        #     task_id = "BlockPicking-Symm-Dict"
        # elif task == 'block_pull':
        #     task_id = "BlockPulling-Symm-Dict"
        client = gym.make(task_id, rendering=pybullet_gui)
        client.unwrapped._obs_dict = True
        super().__init__(client)
        self._new_obs_shape = None
        self._oracle_rng = np.random.RandomState(0)
        self._eps_int = 0

    def reset(self):
        self.timestep = 0
        self._eps_int+=1
        obs = self.client.reset()
        obs = self._process_obs(obs)
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

        return obs, reward, done, info

    def render(self, mode):  # ['human', 'rgb_array', 'mask_array']
        return self.client.render(mode=mode)

    def get_oracle_action(self, obs=None):
        return self.client.query_expert(self._eps_int)

    def _process_obs(self, _obs):
        new_obs = _obs.copy()
        new_obs["image"] = _obs['image']
        obs_t = np.transpose(_obs['image'], axes=[1, 2, 0])
        obs_t = np.concatenate(
            [obs_t, np.zeros(obs_t.shape[:2]+(1,), dtype=np.uint8)], axis=2)
        new_obs['image_new'] = np.uint8(obs_t*255)  # real depth to depth image
        for k, v in new_obs["depth"].items():
            new_obs["depth"][k][np.logical_not(new_obs["mask"][k])] = 1
        new_obs["depthR"] = deepcopy(_obs["depth"])
        new_obs['depth'] = {k: np.uint8(
            scale_arr(v, 0, 1, 0, 255)) for k, v in _obs['depth'].items()}
        gripper_d = np.mean(new_obs["depthR"]["gripper"][new_obs["mask"]["gripper"]])
        # if "object2" in new_obs["depthR"]:
        object_d = np.mean(new_obs["depthR"]["object2"][new_obs["mask"]["object2"]])
        # else:
        #     object_d = 0
        new_obs['z_distance'] = gripper_d - object_d
        return new_obs

    @property
    def observation_space(self):
        if self._new_obs_shape is None:
            obs = self.reset()
            self._new_obs_shape = obs['image'].shape
        return gym.spaces.Box(0, 255, self._new_obs_shape,
                              dtype=np.uint8)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.client.seed(seed)
        self.client.core_env.pose_rng(seed)
        self._oracle_rng = np.random.RandomState(seed)

    @property
    def instrinsic_K(self):
        proj = self.client.get_projection_matrix()
        K = projection_matrix_to_K(proj, image_size=84)
        return K

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        return getattr(self.client, name)
