from rgbd_sym.api import make_env
from rgbd_sym.api import make_env
from rgbd_sym.env.wrapper import Visualizer, ActionOracle
import argparse
import numpy as np
from rgbd_sym.tool.sym import get_random_transform_params, perturb

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--sym-deg", type=int, default=0
                    )

args = parser.parse_args()
env, env_config = make_env(tags=[], seed=args.seed)
obs = env.reset()
env = Visualizer(env, update_hz=-1, vis_tag=["rgb", "obs"], keyboard=True)
done = False

theta, trans, pivot = get_random_transform_params(obs['image'].shape)
theta = np.deg2rad(args.sym_deg)

action = np.array([0, 1.0, 0, 0, 0])
while not done:
    obs_old = obs["image"]
    obs_new, _, action_new, _ = perturb(
        obs["image"],
        None,
        action,
        theta,
        trans,
        pivot,
        set_theta_zero=False,
        set_trans_zero=True,
        action_only=False,
    )
    obs, reward, done, info = env.step(action_new)
    img = {"rgb": obs_old, "obs": obs_new}
    img_break = env.cv_show(imgs=img)
