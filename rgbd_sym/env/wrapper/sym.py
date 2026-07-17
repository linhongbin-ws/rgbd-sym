from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np
import gym
from rgbd_sym.tool.sym import generate_sym3
from rgbd_sym.tool.sym_v2 import generate_sym_v2
from copy import deepcopy as cp

class Sym(BaseWrapper):


    def __init__(self, env,
                 dummy_env,
                 sym_trans_z_low=0.75,
                 sym_trans_z_high=1,
                 sym_trans_r_low=0.75,
                 sym_trans_r_high=1,
                 sym_trans_rot_low=0,
                 sym_trans_rot_high=1,
                 sym_rot_low=0,
                 sym_rot_high=0.3,
                 sym_aug_new_eps=12,
                 mea_version='v1',
                 mea_v2_mode='global',
                 mea_v2_max_angle=2 * np.pi,
                 mea_v2_approach_max_angle=None,
                 mea_v2_action_sign=-1.0,
                 mea_v2_anchor='origin',
                 mea_v2_reflect_prob=0.0,
                 mea_v2_context_channel=False,
                 **kwargs,
                 ):
        super().__init__(env)
        self._dummy_env = dummy_env
        self._sym_eps = []
        self._eps_buffer = []
        self._is_sym = True
        self._step = 0
        self._sym_action = np.zeros(5)

        # mea_v2 (context-conditioned aug). 'v1' -> unchanged generate_sym3 path.
        self._mea_version = mea_version
        self._mea_v2_mode = mea_v2_mode
        self._mea_v2_max_angle = mea_v2_max_angle
        self._mea_v2_approach_max_angle = mea_v2_approach_max_angle
        self._mea_v2_action_sign = mea_v2_action_sign
        self._mea_v2_anchor = mea_v2_anchor
        self._mea_v2_reflect_prob = mea_v2_reflect_prob
        self._mea_v2_context_channel = mea_v2_context_channel

        self._sym_trans_z_low = sym_trans_z_low
        self._sym_trans_z_high = sym_trans_z_high 

        self._sym_trans_r_low = sym_trans_r_low 
        self._sym_trans_r_high = sym_trans_r_high 

        self._sym_trans_rot_low = sym_trans_rot_low 
        self._sym_trans_rot_high = sym_trans_rot_high

        self._sym_rot_low = sym_rot_low 
        self._sym_rot_high = sym_rot_high  

        self._sym_aug_new_eps = sym_aug_new_eps

        self._gt_steps = 0
        self._gt_rollouts = 0

    def reset(self,):
        self._step = 0
        self._gt_rollouts+=1
        if len(self._sym_eps) != 0 and self._is_sym:
            obs,reward, done, info, action = self._sym_eps[0][self._step]
            obs['sym_state'] = 1
            self._sym_state = obs['sym_state']
            self._sym_action = obs['sym_action']
            return obs
        else:
            obs = self.env.reset()
            reward, done, info, action = None, None, None, 0
            if self._is_sym:
                self._eps_buffer = []
                self._eps_buffer.append((cp(obs), cp(reward), cp(done), cp(info), cp(action)))
            obs['sym_action'] = action
            obs['sym_state'] = 0
            self._sym_state = obs['sym_state']
            self._sym_action = obs['sym_action']
            return obs



    def step(self, action):
        self._step += 1
        if len(self._sym_eps) != 0 and self._is_sym:
            obs,reward, done, info, action = self._sym_eps[0][self._step]
            obs['sym_state'] = 1
            self._sym_state = obs['sym_state']
            self._sym_action = obs['sym_action']
            if done:
                self._sym_eps = self._sym_eps[1:]
        else:
            obs, reward, done, info = self.env.step(action)
            self._gt_steps+=1
            obs['sym_action'] = action
            obs['sym_state'] = 0
            self._sym_state = obs['sym_state']
            self._sym_action = obs['sym_action']
            if self._is_sym:
                self._eps_buffer.append((cp(obs), cp(reward), cp(done), cp(info), cp(action)))
                if done:
                    self._on_end_gt_eps()
            return obs, reward, done, info


        return obs, reward, done, info


    
    def _on_end_gt_eps(self,):
        obss_origin = [v[0] for v in self._eps_buffer]
        actions_origin = [v[4] for v in self._eps_buffer]
        actions_origin = actions_origin[1:]



        new_sym_obss = []
        new_sym_actionss = []
        for _ in range(self._sym_aug_new_eps):
            if self._mea_version == 'v2':
                new_sym_obs, new_sym_actions = generate_sym_v2(
                    obss_origin, actions_origin,
                    dummy_env=self._dummy_env,
                    mode=self._mea_v2_mode,
                    max_angle=self._mea_v2_max_angle,
                    approach_max_angle=self._mea_v2_approach_max_angle,
                    action_sign=self._mea_v2_action_sign,
                    anchor=self._mea_v2_anchor,
                    reflect_prob=self._mea_v2_reflect_prob,
                    context_channel=self._mea_v2_context_channel)
            else:
                sym_trans_z = np.random.uniform(self._sym_trans_z_low, self._sym_trans_z_high)
                sym_trans_r = np.random.uniform(self._sym_trans_r_low, self._sym_trans_r_high)
                sym_trans_rot = np.random.uniform(self._sym_trans_rot_low, self._sym_trans_rot_high * 2 * np.pi)
                sym_rot = np.random.uniform(self._sym_rot_low, self._sym_rot_high)
                new_sym_obs, new_sym_actions = generate_sym3(obss_origin,actions_origin,
                                                            sym_trans_z=sym_trans_z,
                                                            sym_trans_r=sym_trans_r,
                                                            sym_trans_rot=sym_trans_rot,
                                                            sym_rot=sym_rot,
                                                            dummy_env=self._dummy_env)
            new_sym_obss.append(new_sym_obs)
            new_sym_actionss.append(new_sym_actions)

        # new_sym_obss, new_sym_actionss = generate_sym3(obss_origin,actions_origin, dummy_env=self._dummy_env,
        #                                                sym_start_step=0)
        for i in range(len(new_sym_obss)):
            _ep = []
            for j in range(len(self._eps_buffer)):
                obs = cp(new_sym_obss[i][j])
                reward = cp(self._eps_buffer[j][1])
                done = cp(self._eps_buffer[j][2])
                info = cp(self._eps_buffer[j][3])
                action = np.zeros(5) if j == 0 else cp(new_sym_actionss[i][j-1])
                obs['sym_action']  = action
                _ep.append((obs, reward, done, info, action,))
            self._sym_eps.append(_ep)
        self._eps_buffer = []

    def set_sym(self, is_sym):
        self._is_sym = is_sym

    @property
    def is_enable_sym(self):
        return self._is_sym


    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        obs['sym_action'] = gym.spaces.Box(low=-1,
                                          high=1, shape=(5,), dtype=float)
        obs['sym_state'] = gym.spaces.Box(low=0,
                                          high=1, shape=(1,), dtype=float)
        return gym.spaces.Dict(obs)
    @property
    def gt_steps(self):
        return self._gt_steps

    @property
    def gt_rollouts(self):
        return self._gt_rollouts
    
    @property
    def sym_state(self):
        return self._sym_state

    @property
    def sym_action(self):
        return self._sym_action

    def mea_rollouts(self, eps):
        self._sym_aug_new_eps = eps
    
    # @property
    # def sym_aug_new_eps(self):
    #     return self._sym_aug_new_eps
    
    # @sym_aug_new_eps.setter
    # def sym_aug_new_eps(self, eps):
    #     self._sym_aug_new_eps = eps


    # @property
    # def seed(self):
    #     return self._seed

    # @seed.setter
    # def seed(self, seed):
    #     self._seed = seed
    #     self.client.seed(self._seed)