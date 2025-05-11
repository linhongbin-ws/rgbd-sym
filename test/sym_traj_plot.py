import sys
sys.path.append("../ext/equi-rl-for-pomdps")
from rgbd_sym.api import make_env
import argparse
import numpy as np
from rgbd_sym.tool.plt import use_backend, plot_img, plot_traj
import matplotlib.pyplot as plt
from rgbd_sym.tool.sym import local_sym_step, get_sym_params, generate_sym2, actions2Ts
from rgbd_sym.tool.depth import get_intrinsic_matrix, projection_matrix_to_K
import time
from copy import deepcopy
from utils.helpers import get_random_transform_params, perturb

size = 84

parser = argparse.ArgumentParser()
parser.add_argument('--env-tag', type=str, nargs='+', default=['block_pick'])
parser.add_argument('--traj-num', type=int, default=None)
parser.add_argument('--radius', type=float, default=None)
parser.add_argument('--height', type=float, default=None)
parser.add_argument('--screw', type=float, default=None)
parser.add_argument('--rot', type=float, default=None)

parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


env, env_config = make_env(tags=args.env_tag, seed=0)
# env = ActionOracle(env, device="script")
for i in range(args.seed):
    obs = env.reset()

obs = env.reset()
proj_matrix = env.get_projection_matrix()
K = projection_matrix_to_K(proj_matrix, image_size=size)
print("K", K)

sym_args = env.sym_args
if args.traj_num is not None:
    sym_args['traj_nums'] = args.traj_num
if args.radius is not None:
    sym_args['radius_ratio'] = args.radius
if args.height is not None:
    sym_args['height_ratio'] = args.height
if args.screw is not None:
    sym_args['screw_angle'] = args.screw
if args.rot is not None:
    sym_args['rot_noise_ratio'] = args.rot

# sym_args['radius_ratio'] = 1
# sym_args['height_ratio'] = 1
# sym_args['screw_angle'] =  0
# generate ground truth trajectory
origin_actions = []
obss_origin = [obs]
done = False
while not done:
    action = env.get_oracle_action()
    origin_actions.append(action)
    obs, reward, done, info = env.step(action)
    obss_origin.append(obs)









new_obss, new_actionss = generate_sym2(obss_origin, origin_actions, 
              **sym_args)
for i in range(len(origin_actions)):
    print(f"step{i}")
    print("origin action", origin_actions[i])
    print("retrace action", new_actionss[0][i])


imgss = []
imgss.append([{"image": v['image'][0,:,:], "title": f"GT step {i}"} for i, v in enumerate(obss_origin)])
for new_obs in new_obss:
    imgss.append([{"image": v['image'][0,:,:], "title": f"Aug step {i}"} for i, v in enumerate(new_obs)])
plt.rcParams['figure.figsize'] = [30, 10]
plot_img(imgss)



points_mats = []
trajTs = actions2Ts(
    [-a for a in reversed(origin_actions)],
    action_delta_pos=sym_args["action_delta_pos"],
    action_delta_rot=sym_args["action_delta_rot"],
)
points_mats = []
pc_mat = {}
pc_mat["mat"] = np.array([[t[0][3], t[1][3], t[2][3]] for t in trajTs])
pc_mat["linewidth"] = 4
pc_mat["alpha"] = 1
pc_mat["label"] = f"original traj"
pc_mat["color"] = "k"
points_mats.append(pc_mat)

for i, new_actions in enumerate(new_actionss):
    new_trajTs = actions2Ts(
    [-a for a in reversed(new_actions)],
    action_delta_pos=sym_args["action_delta_pos"],
    action_delta_rot=sym_args["action_delta_rot"],
    )
    pc_mat = {}
    pc_mat["mat"] = np.array([[t[0][3], t[1][3], t[2][3]] for t in new_trajTs])
    pc_mat["linewidth"] = 4
    pc_mat["alpha"] = 0.5
    pc_mat["label"] = f"traj {i+1}"
    points_mats.append(pc_mat)   
plot_traj(points_mats, elev=45, azim=135, roll=0)





image_size = (84,84)



aug_image = [obss_origin[0]['image'][0,:,:]]
for o in new_obss:
    aug_image.append(o[0]['image'][0,:,:])

imgs = []
imgss = []
for i in range(6):
    imgs = []
    theta, trans, pivot = get_random_transform_params(image_size)
    for o in aug_image:
        print("theta, trans, pivot,",theta, trans, pivot)
        new_o, _, _, _ = perturb(o.copy(),
                                        o.copy(),
                                        np.zeros(2),
                                        theta, trans, pivot,
                                        set_trans_zero=True)
        imgs.append(new_o)
    imgss.append(imgs)
plot_img(imgss)


