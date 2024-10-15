from rgbd_sym.api import make_env
from rgbd_sym.tool.common import getT, TxT, scale_arr
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
from rgbd_sym.tool.depth import get_intrinsic_matrix, depth_image_to_point_cloud, pointclouds2occupancy, occup2image
from rgbd_sym.tool.sym import local_depth_transform, obs_transform
import numpy as np
import cv2
from copy import deepcopy
env1, env_config = make_env(tags=['no_clutch'], seed=0)
env2, env_config = make_env(tags=['no_clutch'], seed=0)
obs1 = env1.reset()
# obs1 = env1.reset()
obs2 = env2.reset()
print(env1.action_space)
action1 = np.zeros(5)
action1[1] = 1
action2 = np.zeros(5)
action2[2] = 1

obs1,_,_,_ = env1.step(action1)
obs2,_,_,_ = env2.step(action2)

obss = [obs1, obs2]
new_obss = []

transform_dict= {}
transl_scale = 0.08 * 0.2
transform_dict['gripper'] = getT([0,0,0], [0,0,0], rot_type="euler", euler_Degrees=True)
transform_dict['object1'] = getT([-transl_scale*np.sin(np.deg2rad(45)),-transl_scale*np.sin(np.deg2rad(45)),0], [0,0,0], rot_type="euler")
transform_dict['object2'] = transform_dict['object1'].copy()

new_obss = []
for i, obs in enumerate(obss):
    _m = None
    for k,v in obs['mask'].items():
        _m = np.logical_or(_m, v) if _m is not None else v
    obs['depthReal'][np.logical_not(_m)] = 1
    obs['depth'] = scale_arr(obs['depthReal'], 0,1, 0 ,255)
    new_obs = deepcopy(obs)
    new_obs['depth'], new_obs['mask'] = obs_transform(obs['depth'], obs['mask'], transform_dict)
    new_obss.append(new_obs)

    imgs = [v for k,v in obs["mask"].items()]
    imgs.append(obs['image'])
    imgs.append(obs['rgb'])
    for i, img in enumerate(imgs):
        ax = subplot(1, len(imgs), i+1)
        imshow(img)
        plt.colorbar()
        ax.set_title(f"obs {i+1}")
    show()


im1 = [obs['depth'] for obs in obss]
im2 = [obs['depth'] for obs in new_obss]
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