from rgbd_sym.api import make_env
import argparse
import numpy as np
from rgbd_sym.tool.plt import plot_img, get_backend, use_backend

from rgbd_sym.tool.sym import local_sym_step
from rgbd_sym.tool.depth import get_intrinsic_matrix, projection_matrix_to_K
import time
use_backend('tkagg')


sym_step_idx = 5
size = 84

args = {}

args['action_delta_pos'] = 0.05
args['action_delta_rot'] = np.pi / 8
args['pc_x_center'] = 0.0
args['pc_y_center'] = 0.0
args['pc_z_center'] = 0.75
args['pc_range'] = 0.8
args['voxel_res'] = 84
args['depth_real_min'] = 0
args['depth_real_max'] = 1
args['depth_upsample'] = 6
args['out_image_type'] = 'depth'
args['out_background_encoding'] = 255


env, env_config = make_env(tags=['block_pull', "no_clutch"], seed=0)
obs = env.reset()
proj_matrix = env.get_projection_matrix()
K = projection_matrix_to_K(proj_matrix, image_size=size)
print("K", K)


# generate ground truth trajectory
origin_actions = []
imgs1_meta = [obs]
done = False
while not done:
    action = env.pull_movable()
    origin_actions.append(action)
    obs, reward, done, info = env.step(action)
    imgs1_meta.append(obs)
origin_depth_image_traj = []
for i in range(len(imgs1_meta)):
    origin_depth_image_traj.append(local_sym_step(imgs1_meta[i]['depth'],
                                                  imgs1_meta[i]['mask'], [],
                                                  K=K,
                                                  **args)[0])


# generate symmetric trajectry with backtrace
start_depth_dict = imgs1_meta[sym_step_idx]['depth']
start_mask_dict = imgs1_meta[sym_step_idx]['mask']
sym_actions = [a for a in origin_actions[:sym_step_idx]]
start = time.time()

sym_depth_image_traj = local_sym_step(
    start_depth_dict, 
    start_mask_dict, 
    sym_actions, 
    K=K, 
    reverse=True,
    **args)

print("local_sym_step() elspase time: ", time.time()- start)
sym_depth_image_traj.extend(origin_depth_image_traj[sym_step_idx+1:])
sym_actions.extend(origin_actions[sym_step_idx:])

sym_actions = [a for a in origin_actions[:sym_step_idx]]
sym_actions[0][1] = -1
sym_actions[1][1] = -1
sym_actions[2][1] = 1
sym_actions[3][1] = 1

sym_depth_image_traj2 = local_sym_step(
    start_depth_dict, 
    start_mask_dict, 
    sym_actions, 
    K=K, 
    reverse=True,
    **args)
sym_depth_image_traj2.extend(origin_depth_image_traj[sym_step_idx+1:])
sym_actions.extend(origin_actions[sym_step_idx:])

# ground truth env with sym actions
env, env_config = make_env(tags=['block_pull', "no_clutch"], seed=0)
origin_actions = []
obs = env.reset()
imgs2_meta = [obs]
for action in sym_actions:
    obs, reward, done, info = env.step(action)
    imgs2_meta.append(obs)
gt_env_traj_with_sym_actions = []
for i in range(len(imgs2_meta)):
    gt_env_traj_with_sym_actions.append(local_sym_step(imgs2_meta[i]['depth'],
                                                  imgs2_meta[i]['mask'], [],
                                                  K=K,
                                                  **args)[0])

# plots
imgss = []
imgss.append([{"image": v, "title": "GT env trajectory"} for v in origin_depth_image_traj])
imgss.append([{"image": v, "title": "Back trace with the same actions"} for v in sym_depth_image_traj])
imgss.append([{"image": v, "title": "Back trace with action seq 1"} for v in sym_depth_image_traj2])
imgss.append([{"image": v, "title": "GT env trajectory with action seq 1"} for v in gt_env_traj_with_sym_actions])
# imgss.append(sym_depth_image_traj)
# imgss.append(sym_depth_image_traj2)
# imgss.append(gt_env_traj_with_sym_actions)
use_backend('tkagg')
plot_img(imgss)
print("backend:", get_backend())
