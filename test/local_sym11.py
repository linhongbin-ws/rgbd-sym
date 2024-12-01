from rgbd_sym.api import make_env
import argparse
import numpy as np
# from rgbd_sym.tool.sym import get_random_transform_params, perturb
from utils.helpers import get_random_transform_params, perturb
from rgbd_sym.tool.plt import plot_img, get_backend, use_backend

from rgbd_sym.tool.sym import local_sym_step
from rgbd_sym.tool.depth import get_intrinsic_matrix, projection_matrix_to_K
use_backend('tkagg')


env, env_config = make_env(tags=['block_pull', "no_clutch"], seed=0)
obs = env.reset()
# env = Visualizer(env, update_hz=-1, vis_tag=["rgb", "obs"], keyboard=True)
done = False

print(env.get_projection_matrix())

import pybullet as pb
proj_matrix = env.get_projection_matrix()
size = 84
K = projection_matrix_to_K(proj_matrix, image_size=size)
print("K", K)


origin_actions = []
steps = 3
imgs1_meta = [obs]
done = False

while not done:
    action = env.pull_movable()
    origin_actions.append(action)
    obs, reward, done, info = env.step(action)
    imgs1_meta.append(obs)

origin_depth_image_traj = []
for i in range(len(imgs1_meta)):
    origin_depth_image_traj.append( local_sym_step(imgs1_meta[i]['depth'], imgs1_meta[i]['mask'], [],K=K)[0])



start_depth_dict = imgs1_meta[0]['depth']
start_mask_dict = imgs1_meta[0]['mask']
actions = [a for a in origin_actions]
delta_rot = 0.3
for i in range(len(actions)):
    actions[i][4] = np.clip(actions[i][4]+delta_rot, -1, 1)
    # actions[i][3] = -1
print("actions len", len(actions))
depth_image_traj = local_sym_step(start_depth_dict, start_mask_dict, actions,K=K)


# start_depth_dict = imgs1_meta[-1]['depth']
# start_mask_dict = imgs1_meta[-1]['mask']
# actions = [ action for i in range(steps)]
# depth_image_traj2 = local_sym_step(start_depth_dict, start_mask_dict, actions, K=K2, reverse=True, debug=False)



 
imgs1 = [im['image'][0,:,:]*255 for im in imgs1_meta]
imgs2 = depth_image_traj
img3 = origin_depth_image_traj
aug_imgss = [imgs1,imgs2, img3]
print(depth_image_traj)


# imgs3 = depth_image_traj2
# imgs4 = [im['mask']['object2']for im in imgs1_meta]
# imgs5 = [im['depthR']['object2']for im in imgs1_meta]
# imgs6 = [im['rgb'] for im in imgs1_meta]
# aug_imgss = [imgs1, imgs2, imgs3,imgs4,imgs5,imgs6]

use_backend('tkagg')
plot_img(aug_imgss)
print("backend:", get_backend())
