import numpy as np
import cv2
from rgbd_sym.tool.common import scale_arr, getT, TxT
from rgbd_sym.tool.img_tool import bool_resize
import numpy as np
from scipy.ndimage import affine_transform
from copy import deepcopy
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
from rgbd_sym.tool.depth import get_intrinsic_matrix, occup2image, scale_K
# from rgbd_sym.tool.depth import depth_image_to_point_cloud
# from rgbd_sym.tool.depth import pointclouds2occupancy
from rgbd_sym.tool.o3d import depth_image_to_point_cloud
from rgbd_sym.tool.o3d import pointclouds2occupancy
from copy import deepcopy


def get_sym_params(env_name):
    if env_name == "block_pull":
        params = {}
        params['action_delta_pos'] = 0.05
        params['action_delta_rot'] = np.pi / 8
        params['pc_x_center'] = 0.0
        params['pc_y_center'] = 0.0
        params['pc_z_center'] = 0.75
        params['pc_range'] = 0.8
        params['voxel_res'] = 84
        params['depth_real_min'] = 0
        params['depth_real_max'] = 1
        params['depth_upsample'] = 6
        params['out_image_type'] = 'depth'
        params['out_background_encoding'] = 255

    else:
        raise NotImplementedError
    return params



def local_depth_transform(depth_image, mask_dict,
                          K, depth_real_min, depth_real_max,
                          transform_dict,
                          pc_x_center,
                          pc_y_center,
                          pc_z_center,
                          pc_range,
                          voxel_res,
                          out_image_type='depth',
                          out_background_encoding=255,
                          depth_upsample=1,
                          debug=False,
                          ):
    depth = cv2.resize(depth_image,
                       (int(depth_image.shape[0]*depth_upsample),
                        int(depth_image.shape[1]*depth_upsample),),
                       interpolation=cv2.INTER_NEAREST)
    _mask_dict = {k: bool_resize(v, depth.shape, reverse=True)
                  for k, v in mask_dict.items()}
    depth_real = scale_arr(np.float32(
        depth), 0, 255, depth_real_min, depth_real_max)  # depth image to depth
    encode_mask = np.zeros(depth.shape, dtype=np.uint8)

    masks = []
    encode_id = {}
    m_id = 0
    for k, v in _mask_dict.items():
        m_id += 1
        masks.append(k)
        encode = m_id
        encode_mask[v] = encode  # background 0, other mask key 1, 2, 3 ...
        encode_id[k] = encode

    scale = 1
    pose = np.eye(4)
    rgb = np.zeros(depth.shape + (3,), dtype=np.uint8)
    new_K = scale_K(K,
                    depth.shape[0]/depth_image.shape[0],
                    depth.shape[1]/depth_image.shape[1], )
    points = depth_image_to_point_cloud(
        rgb, depth_real, scale, new_K, pose, encode_mask=encode_mask, tolist=False
    )
    # print("point range", np.min(points[:,:3], axis=0),np.max(points[:,:3], axis=0,))

    for k, v in transform_dict.items():
        pc_idx = points[:, 6] == encode_id[k]
        ones = np.ones((points[pc_idx, :].shape[0], 1))
        P = np.concatenate((points[pc_idx, :3], ones), axis=1)
        points[pc_idx, :3] = np.matmul(P, np.transpose(v))[:, :3]

    points = points[points[:, 6] != 0, :]  # remove background
    occ_mat = pointclouds2occupancy(
        points,
        occup_h=voxel_res,
        occup_w=voxel_res,
        occup_d=voxel_res,
        pc_x_min=pc_x_center - pc_range/2,
        pc_x_max=pc_x_center + pc_range/2,
        pc_y_min=pc_y_center - pc_range/2,
        pc_y_max=pc_y_center + pc_range/2,
        pc_z_min=pc_z_center - pc_range/2,
        pc_z_max=pc_z_center + pc_range/2,
    )
    z, z_mask = occup2image(occ_mat, image_type=out_image_type,
                            background_encoding=out_background_encoding)
    if debug:
        px, py, pz = occup2image(
            occ_mat, image_type='projection', background_encoding=out_background_encoding)
        from rgbd_sym.tool.plt import plot_img
        plot_img([[px, py, pz, depth_real]])
    del occ_mat
    s = depth_image.shape
    z = cv2.resize(z, (s[0], s[1]), interpolation=cv2.INTER_NEAREST)
    z_mask = bool_resize(z_mask, (s[0], s[1]),
                         method=cv2.INTER_NEAREST, reverse=True)
    return z, z_mask


def obs_transform(obs_depths, 
                  obs_masks, 
                  transform_dict,
                  K,
                  pc_x_center,
                  pc_y_center,
                  pc_z_center,
                  pc_range,
                  voxel_res,
                  depth_real_min,
                  depth_real_max,
                  depth_upsample, 
                  out_image_type,
                  out_background_encoding,
                  debug=False,
                  ):
    _obs_depths = deepcopy(obs_depths)
    _obs_masks = deepcopy(obs_masks)
    new_obs_depth = {}
    new_obs_masks = {}
    for k, v in _obs_depths.items():
        depth_new, mask_new = local_depth_transform(v,
                                                    mask_dict={
                                                        k: _obs_masks[k]},
                                                    transform_dict={
                                                        k: transform_dict[k]},
                                                    K=K,
                                                    depth_real_min=depth_real_min,
                                                    depth_real_max=depth_real_max,
                                                    pc_x_center=pc_x_center,
                                                    pc_y_center=pc_y_center,
                                                    pc_z_center=pc_z_center,
                                                    pc_range=pc_range,
                                                    voxel_res=voxel_res,
                                                    out_image_type=out_image_type,
                                                    out_background_encoding=out_background_encoding,
                                                    depth_upsample=depth_upsample,
                                                    debug=debug)

        new_obs_depth[k] = depth_new
        new_obs_masks[k] = mask_new

    if debug:
        from rgbd_sym.tool.plt import plot_img
        img1 = [v for _, v in obs_masks.items()]
        img2 = [v for _, v in new_obs_depth.items()]
        img3 = [v for _, v in new_obs_masks.items()]
        plot_img([img1, img2, img3])

    return new_obs_depth, new_obs_masks


def action2transformdict(action, delta_pos, delta_rot, reverse=False, ):
    transform_dict = {}
    # 'dpos': 0.05, 'drot': np.pi/8
    # rot_scale = np.pi/8
    # transl_scale = 0.05 * 0.27
    sign = -1 if reverse else 1
    transform_dict['gripper'] = getT(
        [0, 0, 0], [0, 0, action[4]*sign*delta_rot], rot_type="euler", euler_Degrees=False)
    transform_dict['object1'] = getT([-delta_pos*action[1]*sign,
                                      -delta_pos*action[2]*sign,
                                      delta_pos*action[3]*sign,],
                                     [0, 0, 0],
                                     rot_type="euler")
    transform_dict['object2'] = transform_dict['object1'].copy()
    return transform_dict


def local_sym_step(start_depth_dict,
                   start_mask_dict,
                   actions,
                   K,
                   action_delta_pos,
                   action_delta_rot,
                   pc_x_center,
                   pc_y_center,
                   pc_z_center,
                   pc_range,
                   voxel_res,
                   depth_real_min,
                   depth_real_max,
                   depth_upsample=6,
                   out_image_type='depth',
                   out_background_encoding=255,
                   reverse=False,
                   debug=False):
    T_dict = None
    depth_dict_traj = []
    mask_dict_traj = []
    
    _actions = [v for v in reversed(actions)] if reverse else actions
    _actions = [np.zeros(5)] + _actions
    for action in _actions:
        delta_T_dict = action2transformdict(action,
                                            reverse=reverse,
                                            delta_pos=action_delta_pos,
                                            delta_rot=action_delta_rot)
        if T_dict is None:
            T_dict = delta_T_dict
        else:
            T_dict = {k: TxT([delta_T_dict[k], v]) for k, v in T_dict.items()}

        _depth, _mask = obs_transform(start_depth_dict,
                                      start_mask_dict,
                                      T_dict,
                                      K=K,
                                      pc_x_center=pc_x_center,
                                      pc_y_center=pc_y_center,
                                      pc_z_center=pc_z_center,
                                      pc_range=pc_range,
                                      voxel_res=voxel_res,
                                      depth_real_min=depth_real_min,
                                      depth_real_max=depth_real_max,
                                      depth_upsample=depth_upsample,
                                      out_image_type=out_image_type,
                                      out_background_encoding=out_background_encoding,
                                      debug=debug
                                      )
        depth_dict_traj.append(_depth)
        mask_dict_traj.append(_mask)

    depth_image_traj = [get_depth_image_from_dict(
        _d) for _d in depth_dict_traj]
    if reverse:
        depth_image_traj = [v for v in reversed(depth_image_traj)]
    return depth_image_traj

def generate_sym(obs, actions,sym_step_idx, **args):
    sym_actions = [a for a in actions[:sym_step_idx]]
    for k in range(len(sym_actions)):
        # sym_actions[0] = np.random.uniform(-1,1)
        sym_actions[k][1] = np.random.uniform(-1,1)
        sym_actions[k][2] = np.random.uniform(-1,1)
        sym_actions[k][3] = np.random.uniform(-1,1)
        sym_actions[k][4] = np.random.uniform(-1,1)

    start_depth_dict = obs[sym_step_idx]['depth']
    start_mask_dict = obs[sym_step_idx]['mask']

    sym_depth_image_traj = local_sym_step(
        start_depth_dict, 
        start_mask_dict, 
        sym_actions,  
        reverse=True,
        **args)
    new_obs = deepcopy(obs)
    for i, d in enumerate(sym_depth_image_traj):
        new_obs[i]['image'] = np.stack([d / 255, np.zeros(d.shape, dtype=float)], axis=0)
    
    return new_obs

def get_depth_image_from_dict(depth_dict):
    depths = [v for k, v in depth_dict.items()]
    new_obs_depth = np.min(np.stack(depths, axis=0), axis=0)
    return new_obs_depth
