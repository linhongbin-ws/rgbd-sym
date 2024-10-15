from rgbd_sym.api import make_env
from rgbd_sym.tool.common import getT, TxT, scale_arr
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
from rgbd_sym.tool.depth import get_intrinsic_matrix, depth_image_to_point_cloud, pointclouds2occupancy, occup2image
from rgbd_sym.tool.sym import local_depth_transform
import numpy as np
import cv2
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
for i, obs in enumerate(obss):
    _m = None
    for k,v in obs['mask'].items():
        _m = np.logical_or(_m, v) if _m is not None else v
    obs['depthReal'][np.logical_not(_m)] = 1
    obs['depth'] = scale_arr(obs['depthReal'], 0,1, 0 ,255)


fov = 45
gripper_project_offset = 0.2 # gripper is z zero, so projection is not in FOV45, we need to somehow recover
ws_scale = 0.08 
x_offset = +0.00
y_offset = -0.00
z_offset = 0.06
z_scale=12
pc_x_min=-ws_scale+x_offset
pc_x_max=ws_scale+x_offset
pc_y_min=-ws_scale+y_offset
pc_y_max=ws_scale+y_offset
pc_z_min=-ws_scale+z_offset
pc_z_max=ws_scale*z_scale+z_offset
occup_h=70
occup_w=70
occup_d=70
K = get_intrinsic_matrix(obs['depth'].shape[0], obs['depth'].shape[1], fov=45)

transform_dict= {}
transl_scale = ws_scale * 0.2
transform_dict['gripper'] = getT([0,0,0], [0,0,0], rot_type="euler", euler_Degrees=True)
transform_dict['object1'] = getT([-transl_scale*np.sin(np.deg2rad(45)),-transl_scale*np.sin(np.deg2rad(45)),0], [0,0,0], rot_type="euler")
transform_dict['object2'] = transform_dict['object1'].copy()

ks = ["object1","object2","gripper"] # need to put "gripper" at the end so that gripper always on the top of depth image
_depth = obss[0]['depth']
_mask = {k: obss[0]['mask'][k] for k in ks}
depth_new =  local_depth_transform(_depth,
                        mask_dict=_mask, 
                        transform_dict=transform_dict,
                        K=K, 
                        depth_real_min=0, 
                        depth_real_max=1,
                        gripper_project_offset=gripper_project_offset,
                            pc_x_min=pc_x_min,
                            pc_x_max=pc_x_max,
                            pc_y_min=pc_y_min,
                            pc_y_max=pc_y_max,
                            pc_z_min=pc_z_min,
                            pc_z_max=pc_z_max,
                            occup_h=occup_h,
                            occup_w=occup_w,
                            occup_d=occup_d,
                            background_encoding=255)

# imgs = [obss[0]['mask']['object1'],obss[0]['mask']['object2'],obss[0]['mask']['gripper'], obss[0]['rgb'],obss[0]['depth']]
# for i, img in enumerate(imgs):
#     ax = subplot(1, len(imgs), i+1)
#     imshow(img)
#     plt.colorbar()
#     ax.set_title(f"obs {i+1}")

# show()
imgs = [obs1['depth'], obs2['depth'], depth_new]
for i, img in enumerate(imgs):
    ax = subplot(1, len(imgs), i+1)
    imshow(img)
    plt.colorbar()
    ax.set_title(f"obs {i+1}")

show()