from rgbd_sym.api import make_env
import argparse
import numpy as np
# from rgbd_sym.tool.sym import get_random_transform_params, perturb
from utils.helpers import get_random_transform_params, perturb
from rgbd_sym.tool.plt import plot_img, get_backend, use_backend

from rgbd_sym.tool.sym import local_sym_step

use_backend('tkagg')


env, env_config = make_env(tags=['block_pull', "no_clutch"], seed=0)
obs = env.reset()
# env = Visualizer(env, update_hz=-1, vis_tag=["rgb", "obs"], keyboard=True)
done = False

import pybullet as pb
proj_matrix = pb.computeProjectionMatrixFOV(fov=70,
                                aspect=1,
                                nearVal=0.1,
                                farVal=1)
size = obs['image'].shape[1]
proj_matrix = np.array(list(proj_matrix)).reshape(4,4)
K = np.zeros((3,3))
K[0][0] = proj_matrix[0,0] * size /2
K[1][1] = proj_matrix[1,1] * size /2
K[2][2] = 1
print("K", K)
from rgbd_sym.tool.depth import get_intrinsic_matrix
K2 = get_intrinsic_matrix(width=size,height=size, fov=70)
print("K2", K2)

action = np.array([1, 0.0, 0,-1.0, 0])
steps = 3
print(obs["image"].shape)
imgs1 = [obs["image"]]
imgs1_meta = [obs]
for i in range(steps):
    obs, reward, done, info = env.step(action)
    imgs1.append(obs["image"])
    imgs1_meta.append(obs)




start_depth_dict = imgs1_meta[0]['depth']
start_mask_dict = imgs1_meta[0]['mask']
actions = [ action for i in range(steps)]
depth_image_traj = local_sym_step(start_depth_dict, start_mask_dict, actions,K=K2)


start_depth_dict = imgs1_meta[-1]['depth']
start_mask_dict = imgs1_meta[-1]['mask']
actions = [ action for i in range(steps)]
depth_image_traj2 = local_sym_step(start_depth_dict, start_mask_dict, actions, K=K2, reverse=True, debug=False)



 
imgs1 = [im['image'][0,:,:] for im in imgs1_meta]

imgs2 = depth_image_traj
imgs3 = depth_image_traj2
imgs4 = [im['mask']['object2']for im in imgs1_meta]
imgs5 = [im['depthR']['object2']for im in imgs1_meta]
imgs6 = [im['rgb'] for im in imgs1_meta]
aug_imgss = [imgs1, imgs2, imgs3,imgs4,imgs5,imgs6]

use_backend('tkagg')
plot_img(aug_imgss)
print("backend:", get_backend())
