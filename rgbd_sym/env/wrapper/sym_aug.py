from rgbd_sym.env.wrapper.base import BaseWrapper
from rgbd_sym.tool.sym_tool import get_random_transform_params, perturb
import sys
import numpy as np
from copy import deepcopy

class SymAug(BaseWrapper):

    def __init__(self, 
                 env,
                 sym_num=4,
                 **kwargs):
        super().__init__(env)
        self._sym_num = sym_num
        self._eps_idx = 0
        self._timestep = 0
        self._filling_eps_data = True
        self._empty_eps_data()

    def _empty_eps_data(self):
        self._eps_data = {"reset": None, "step":[],}

    def _transform_obs(self, obs, action=None, action_only=False):
        _action = None if action is None else action[0:2]
        theta, trans, pivot = self._transform_param
        image_new, _, _action, _ = perturb(
            obs["image"].copy(),
            None,
            _action,
            theta,
            trans,
            pivot,
            set_trans_zero=True,
            action_only=action_only,
        )
        obs_new = obs.copy()
        obs_new["image"] = image_new
        if action is None:
            action_new = None
        else:
            action_new = action.copy()
            action_new[0:2] = _action
        return obs_new, action_new

    def reset(self, ):
        self._timestep = 0
        self._eps_idx += 1 
        self._timestep += 1
        self._filling_eps_data = self._eps_data["reset"] is not None

        self._eps_idx = self._eps_idx % (self.SymNum + 1) # prevent idx explode
        if self._check_sym_condition():
            obs = deepcopy(self._eps_data["reset"])
            theta, trans, pivot = get_random_transform_params(obs["image"].shape)
            self._transform_param = (theta, trans, pivot,)
            new_obs, _ = self._transform_obs(obs)
            return new_obs
        else:
            self._empty_eps_data()
            obs =  self.env.reset()
            if self.env.mode =="train": # do not update when evaluate env
                self._eps_data["reset"] = obs
            return obs

    def step(self, action):

        if not self._check_sym_condition():
            step_r =  self.env.step(action)
            if self.env.mode =="train": # do not update when evaluate env
                self._eps_data["step"].append(step_r + (action, ))
            self._timestep += 1
            return step_r
        else:
            _idx = self._timestep - 1
            if _idx >= len(self._eps_data["step"]) - 1:
                _idx = len(self._eps_data["step"]) - 1
                done = True
            else:
                done = False
            # print(_idx, len(self._eps_data["step"]))
            obs, reward, _, info, action_ = deepcopy(self._eps_data["step"][_idx])
            new_obs, new_action = self._transform_obs(obs, action_)
            info['action'] = new_action
            self._timestep += 1
            return new_obs, reward, done, info

    def get_sym_action(self):
        _idx = self._timestep - 1
        if _idx >= len(self._eps_data["step"]) - 1:
            _idx = len(self._eps_data["step"]) - 1
        obs, reward, done, info, action_ = self._eps_data["step"][_idx]
        new_obs, new_action = self._transform_obs(obs, action_, action_only=True)
        return new_action

    @property
    def IsSymEnv(self):
        return self._check_sym_condition()

    def _check_sym_condition(self):
        return (
            self._eps_idx % (self.SymNum + 1) != 1
            and self.env.mode == "train"
            and self._filling_eps_data
        )

    @property
    def SymNum(self):
        return self._sym_num
