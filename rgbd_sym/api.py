import rgbd_sym
from rgbd_sym.env.wrapper import *
from rgbd_sym.env.embodied import *
from rgbd_sym.tool.config import Config, load_yaml
from pathlib import Path
import rgbd_sym.env.embodied as ebd
from rgbd_sym.tool.config import Config


def make_env(task, seed=0, sym=True, obs_type='occup', **kwargs):
    env = PomdpEnv(task = task)
    env = Occup(env)

    dummy_env = DummyEnv(task = task)
    dummy_env = Occup(dummy_env)


    env = Sym(env, dummy_env, **kwargs)
    env = GymRegularizer(env, obs_type = obs_type)
    if sym:
        env.set_sym(True)
    else:
        env.set_sym(False)
    env.seed = seed
    return env