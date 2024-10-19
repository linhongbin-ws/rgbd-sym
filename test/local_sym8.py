from rgbd_sym.api import make_env
from rgbd_sym.tool.common import getT, TxT, scale_arr
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
from rgbd_sym.tool.depth import get_intrinsic_matrix, depth_image_to_point_cloud, pointclouds2occupancy, occup2image
from rgbd_sym.tool.sym import local_depth_transform, obs_transform, traj_transform
from rgbd_sym.tool.img_tool import bool_resize
import numpy as np
import cv2
from copy import deepcopy
env1, env_config = make_env(tags=['no_clutch'], seed=0)
original_obs = env1.reset()
traj_obss = [(original_obs, None)]


# done = False
# show_steps = 5
# while not done:
#     action = env1.get_oracle_action()
#     obs, reward, done, info= env1.step(action)
#     traj_obss.append((obs, action))

action = np.zeros(5)
action[4] = 1
show_steps = 3
for i in range(show_steps):
    obs, reward, done, info= env1.step(action)
    traj_obss.append((obs, action))

def get_depth_image(depth_dict):
    depths = [v for k,v in depth_dict.items()]
    new_obs_depth = np.min(np.stack(depths, axis=0), axis=0)
    return new_obs_depth

im1 = [get_depth_image(obs[0]['depth']) for obs in traj_obss]
im2 = [obs[0]['rgb'] for obs in traj_obss]
imgs = [im1, im2]


def action2transformdict(action, reverse=False):
    transform_dict= {}
    # 'dpos': 0.05, 'drot': np.pi/8
    rot_scale = np.pi/8
    transl_scale = 0.04 * 0.2
    sign = -1 if reverse else 1
    transform_dict['gripper'] = getT([0,0,0], [0,0,action[4]*sign*rot_scale], rot_type="euler", euler_Degrees=False)
    transform_dict['object1'] = getT([-transl_scale*action[1]*sign,
                                      -transl_scale*action[2]*sign,
                                      -transl_scale*action[3]*sign,], 
                                      [0,0,0], 
                                     rot_type="euler")
    transform_dict['object2'] = transform_dict['object1'].copy()
    return transform_dict

traj_obss = traj_obss[-show_steps:]
reverse_traj_obss = [v for v in reversed(traj_obss)]

Ts = [action2transformdict(t[1],reverse=True) for t in reverse_traj_obss]
sim_reverse_traj_obss = traj_transform(traj_obss[-1][0], Ts)
sim_reverse_traj_obss = [traj_obss[-1][0]] + sim_reverse_traj_obss[:-1]
sim_traj_obss = [v for v in reversed(sim_reverse_traj_obss)]

im1 = [get_depth_image(obs[0]['depth']) for obs in traj_obss]
im2 = [get_depth_image(obs['depth']) for obs in sim_traj_obss]
imgs = [im1, im2]

plt_cnt = 0
for im in imgs:
    for i, img in enumerate(im):
        plt_cnt+=1
        ax = subplot(len(imgs), len(im), plt_cnt)
        imshow(img,vmin=0, vmax=100)
        plt.colorbar()
        ax.set_title(f"obs {i+1}")

show()