from rgbd_sym.env.wrapper.base import BaseWrapper
from rgbd_sym.tool.sym_tool import get_random_transform_params, perturb
import sys


class SymAug(BaseWrapper):

    def __init__(self, 
                 env,
                 sym_num=4,
                 **kwargs):
        super().__init__(env)
        self._sym_num = sym_num
        self._eps_idx = 0
        self._eps_data = None
        self._timestep = 0
        self._data_collect_phase= False

    def _empty_eps_data(self):
        self._eps_data = {"reset": None, "step":[],}
        
    
    def _transform_obs(self, obs, action=None):
        _action = None if action is None else action[0:2]
        theta, trans, pivot = self._transform_param
        image_new, _, _action, _  = perturb(obs["image"].copy(),
                                            None,
                                            _action,
                                            theta, trans, pivot,
                                            set_trans_zero=True)
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
        self._eps_idx +=1
        self._data_collect_phase =  self._eps_idx % self._sym_num == 1 or self._eps_data is None
        if self._data_collect_phase:
            self._empty_eps_data()
            obs =  self.env.reset()
            self._eps_data["reset"] = obs
            
            return obs
        
        else:
            obs = self._eps_data["reset"]
            theta, trans, pivot = get_random_transform_params(obs["image"].shape)
            self._transform_param = (theta, trans, pivot,)
            new_obs, _ = self._transform_obs(obs)
            return new_obs
        
    def step(self, action):
        self._timestep+=1
        if self._data_collect_phase:
            step_r =  self.env.step(action)
            self._eps_data["step"].append(step_r + (action, ))
            return step_r
        else:
            # print("xxxxxxxxxxxxxxxxxxxxxxxx")
            # print(self._eps_data["step"][0])
            print(len(self._eps_data["step"]))
            print(self._timestep-1)
            obs, reward, done, info, action_  = self._eps_data["step"][self._timestep-1]
            new_obs, new_action = self._transform_obs(obs, action_)
            info['action'] = new_action
            return new_obs, reward, done, info
        