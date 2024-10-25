from rgbd_sym.env.wrapper.base import BaseWrapper
import sys
import numpy as np

class ActionOracle(BaseWrapper):
    KEYBOARD_MAP = {
        "w": np.array([0,1,0,0,0],dtype=float),
        "s": np.array([0,-1,0,0,0],dtype=float),
        "a": np.array([0,0,1,0,0],dtype=float),
        "d": np.array([0,0,-1,0,0],dtype=float),
        "k": np.array([0,0,0,1,0],dtype=float),
        "i": np.array([0,0,0,-1,0],dtype=float),
        "j": np.array([0,0,0,0,1],dtype=float),
        "l": np.array([0,1,0,0,-1],dtype=float),
        "n": np.array([1,0,0,0,0],dtype=float),
        "b": np.array([-1,0,0,0,0],dtype=float),
    }

    def __init__(self, env,
                 device='keyboard',
                 **kwargs):
        super().__init__(env)
        self._device_type = device
        if device == "ds4":
            from rgbd_sym.tool.ds_util import DS_Controller
            self._device = DS_Controller()
        elif device in ['keyboard', 'script']:
            from rgbd_sym.tool.keyboard import Keyboard
            self._device = Keyboard()
        else:
            raise NotImplementedError

    def get_oracle_action(self):
        if self._device_type in ['keyboard', 'script']:
            while True:
                ch = self._device.get_char()
                # print(ch)
                if ch == 'q':
                    sys.exit(0)
                elif ch in self.KEYBOARD_MAP and self._device_type == "keyboard":
                    return self.KEYBOARD_MAP[ch]
                elif self._device_type == "script":
                    return self.env.get_oracle_action()
                else:
                    continue
