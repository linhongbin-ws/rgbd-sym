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


_mask = {k: obss[0]['mask'][k] for k in ks}

imgs = [v for k,v in _mask.items()]
_u_mask = None
for k, v in _mask.items():
    _u_mask = v if _u_mask is None else np.logical_or(_u_mask, v)
imgs.append(_u_mask)
for i, img in enumerate(imgs):
    ax = subplot(1, len(imgs), i+1)
    imshow(img)
    plt.colorbar()
    ax.set_title(f"obs {i+1}")

show()


_depth_old = {}
for k,v in obss[0]['mask'].items():
    _depth_old[k] = np.ones(obss[0]['depth'].shape, dtype=np.uint8)*255
    _depth_old[k][v] = obss[0]['depth'][v]
imgs_old = [v for k,v in _depth_old.items()]
merge_depth = np.min(np.stack(imgs_old, axis=0), axis=0)
imgs_old.append(merge_depth)
plt_cnt = 0
for i, img in enumerate(imgs_old):
    plt_cnt+=1
    ax = subplot(3, 4, plt_cnt)
    imshow(img,vmin=0, vmax=100)
    plt.colorbar()
    ax.set_title(f"obs {i+1}")


_depth = {}
for k,v in obss[0]['mask'].items():
    union_mask = None
    for mask_k, mask_v in _mask.items():
        if mask_k!=k:
            union_mask = mask_v if union_mask is None else np.logical_or(union_mask,mask_v)
    _obj_mask = np.logical_and(v,np.logical_not(union_mask))
    _depth[k] = np.ones(obss[0]['depth'].shape, dtype=np.uint8)*255
    _depth[k][v] = np.median(obss[0]['depth'][_obj_mask])

imgs = [v for k,v in _depth.items()]
merge_depth = np.min(np.stack(imgs_old, axis=0), axis=0)
imgs.append(merge_depth)
for i, img in enumerate(imgs):
    plt_cnt+=1
    ax = subplot(3, 4, plt_cnt)
    imshow(img,vmin=0, vmax=100)
    plt.colorbar()
    ax.set_title(f"obs {i+1}")




depths = []
for k,v in _depth.items():
    depth_new =  local_depth_transform(v,
                            mask_dict={k:_mask[k]}, 
                            transform_dict={k: transform_dict[k]},
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

    depths.append(depth_new
                )
merge_depth = np.min(np.stack(depths, axis=0), axis=0)
imgs = depths
imgs.append(merge_depth)


for i, img in enumerate(imgs):
    plt_cnt+=1
    ax = subplot(3, 4, plt_cnt)
    imshow(img,vmin=0, vmax=100)
    plt.colorbar()
    ax.set_title(f"obs {i+1}")

show()