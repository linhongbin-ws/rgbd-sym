from rgbd_sym.api import make_env
import argparse
import numpy as np
from rgbd_sym.tool.plt import plot_img, get_backend, use_backend

from rgbd_sym.tool.sym import generate_sym, get_sym_params
from rgbd_sym.tool.depth import get_intrinsic_matrix, projection_matrix_to_K
import time
use_backend('tkagg')


moveable = False

# generate ground truth trajectory
env, env_config = make_env(tags=['block_pull', "no_clutch"], seed=0)
obs = env.reset()
origin_actions = []
imgs1_meta = [obs]
done = False
while not done:
    if moveable:
        action = env.query_expert(1)
    else:
        action = env.query_expert(2)
    origin_actions.append(action)
    obs, reward, done, info = env.step(action)
    imgs1_meta.append(obs)



sym_args = get_sym_params(env_name="block_pull")
sym_args['K'] = env.unwrapped.instrinsic_K
sym_step_idx = 5

new_obs = generate_sym(imgs1_meta, origin_actions,sym_step_idx, **sym_args)

imgss = []
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][0,:,:] for o in imgs1_meta])])
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][1,:,:] for o in imgs1_meta])])
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][0,:,:] for o in new_obs])])
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][1,:,:] for o in new_obs])])
# imgss.append(sym_depth_image_traj)
# imgss.append(sym_depth_image_traj2)
# imgss.append(gt_env_traj_with_sym_actions)
use_backend('tkagg')
plot_img(imgss, big_axes_title=["channel 1","channel 2", "sym channel1","sym channel2"])
print("backend:", get_backend())
