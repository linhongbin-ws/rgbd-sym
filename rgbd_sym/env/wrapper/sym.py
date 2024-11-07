from rgbd_sym.env.wrapper.base import BaseWrapper
import numpy as np

class Sym(BaseWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def step(self, action):
    
    def reset(self):