from rgbd_sym.api import make_env
import argparse
import numpy as np
# from rgbd_sym.tool.sym import get_random_transform_params, perturb
from utils.helpers import get_random_transform_params, perturb
from rgbd_sym.tool.plt import plot_img, get_backend, use_backend






env, env_config = make_env(tags=['block_pull', "no_clutch"], seed=0)
obs = env.reset()
# env = Visualizer(env, update_hz=-1, vis_tag=["rgb", "obs"], keyboard=True)
done = False



action = np.array([0, 1.0, 0, 0, 0])
print(obs["image"].shape)
imgs1 = [obs["image"]]
for i in range(4):
    obs, reward, done, info = env.step(action)
    imgs1.append(obs["image"])


angles = [45,90,135,180]

aug_imgss = []
for a in angles:
    theta, trans, pivot = get_random_transform_params(obs['image'].shape[1:])
    theta = np.deg2rad(a)
    imgs2 = []
    for im in imgs1:
        obs_new = im.copy()
        obs_new[0,:,:], _, action_new, _ = perturb(
            im[0,:,:],
            None,
            action[1:3],
            theta,
            trans,
            pivot,
            set_theta_zero=False,
            set_trans_zero=True,
            # action_only=False,
        )
        imgs2.append(obs_new)
    imgs2 = [im[0,:,:] for im in imgs2]
    aug_imgss.append(imgs2)

imgs1 = [im[0,:,:] for im in imgs1]
aug_imgss = [imgs1] +aug_imgss
# print(imgs1)

use_backend('tkagg')
plot_img(aug_imgss)
print("backend:", get_backend())
