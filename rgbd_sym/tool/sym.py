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

        params['traj_nums']   = 4 
        params['radius_ratio'] = 0.75
        params['height_ratio'] = 1
        params['screw_angle'] =  0
        params['sym_z_distance_thres'] = 0.092

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
                   debug=False,
                   **args,
                   ):
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



def get_depth_image_from_dict(depth_dict):
    depths = [v for k, v in depth_dict.items()]
    new_obs_depth = np.min(np.stack(depths, axis=0), axis=0)
    return new_obs_depth


def Ts2actions(
    origin_T,
    interval_Ts,
    action_delta_pos,
    interval_gripper_states,
    interval_oris,
):
    # print(origin_T)
    actions = []
    prv_T = origin_T[:3,3]
    exceed= False
    for i in range(len(interval_Ts)):
        delta_T = interval_Ts[i][:3, 3] - prv_T
        a_p = scale_arr(delta_T, -action_delta_pos, action_delta_pos, -1, 1)
        a_p[0] = -a_p[0]
        a_p[1] = -a_p[1]
        # a_p[2] = -a_p[2]
        # assert np.all(a_p) >= -1 and np.all(a_p) <=1, a_p
        # assert a_o >= -1 and a_o <= 1, f"ori, {o}, max, {action_delta_rot}, i {i}"
        if  not (np.all(a_p) >= -1 and np.all(a_p) <=1):
            exceed = True
            a_p = np.clip(a_p, -1, 1)
        a = np.array(
            [
                interval_gripper_states[i],
                a_p[0],
                a_p[1],
                a_p[2],
                interval_oris[i],
            ]
        )
        actions.append(a)
        # print(prv_T)
        prv_T = interval_Ts[i][:3, 3]
    return actions, exceed

def actions2Ts(
    actions,
    action_delta_pos,
    action_delta_rot,
):
    trajTs = []
    _actions = actions
    _actions = [np.zeros(5)] + _actions
    T_dict = None
    
    for action in _actions:
        delta_T_dict = action2transformdict(
            action,
            reverse=False,
            delta_pos=action_delta_pos,
            delta_rot=action_delta_rot,
        )
        if T_dict is None:
            T_dict = delta_T_dict
        else:
            T_dict = {k: TxT([delta_T_dict[k], v]) for k, v in T_dict.items()}
        trajTs.append(T_dict)

    trajTs = [v["object2"] for v in trajTs]
    return trajTs

def generate_sym(obs, actions,sym_step_idx, **args):
    sym_actions = [a for a in actions[:sym_step_idx]]
    for k in range(len(sym_actions)):
        # sym_actions[0] = np.random.uniform(-1,1)
        sym_actions[k][1] = np.random.uniform(-1,1)
        sym_actions[k][2] = np.random.uniform(-1,1)
        # sym_actions[k][3] = np.random.uniform(-1,1)
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

def generate_sym2(obs, origin_actions,
                  traj_nums=20, 
                    radius_ratio =1,
                    height_ratio =1,
                    screw_angle = 0,
                    sym_z_distance_thres = 0.092,
                  **args):
    
    for sym_step_idx, _o in enumerate(obs):
        if np.abs(_o["z_distance"]) < sym_z_distance_thres:
            break
    # generate pose of origin trajectory
    trajTs = actions2Ts(
        [-a for a in reversed(origin_actions)],
        action_delta_pos=args["action_delta_pos"],
        action_delta_rot=args["action_delta_rot"],
    )

    # generate poses of symmetric trajectory
    new_trajTss = []
    angles = np.linspace(0, 360, num=traj_nums)
    for traj_idx in range(traj_nums):
        new_trajTs = []
        start_idx = len(trajTs) - sym_step_idx - 1
        screw_cnt = 0
        for idx in range(len(trajTs)):
            if idx <= start_idx:
                new_trajTs.append(trajTs[idx].copy())
            else:
                origin_T = trajTs[start_idx].copy()
                current_T = trajTs[idx].copy()
                current_T[0:3, 3] = current_T[0:3, 3] - origin_T[0:3, 3]
                current_T[0:2, 3] = current_T[0:2, 3] * radius_ratio
                current_T[2, 3] = current_T[2, 3] * height_ratio
                deltaT = getT([0, 0, 0], [0, 0, angles[traj_idx] + screw_cnt * screw_angle], rot_type="euler")
                current_T = TxT([deltaT, current_T])
                current_T[0:3, 3] = current_T[0:3, 3] + origin_T[0:3, 3]
                new_trajTs.append(current_T)
                screw_cnt+=1
        new_trajTss.append(new_trajTs)

    # generate observation for symmetric trajectories
    new_sym_obss = []
    new_sym_actionss = []
    for new_trajTs in new_trajTss:
        _Ts = deepcopy(new_trajTs)
        reverse_actions, exceed = Ts2actions(
            origin_T=_Ts[0],
            interval_Ts=_Ts[1:],
            action_delta_pos=args["action_delta_pos"],
            interval_gripper_states=np.zeros(len(_Ts[1:])),
            interval_oris=np.zeros(len(_Ts[1:])),
        )
        reverse_actions = [a for a in reversed(reverse_actions)]
        reverse_actions_old = [-a for a in origin_actions]
        for k in range(len(reverse_actions_old)):
            if k>= sym_step_idx-1:
                reverse_actions[k][1] = reverse_actions_old[k][1]
                reverse_actions[k][2] =  reverse_actions_old[k][2]
                reverse_actions[k][3] =  reverse_actions_old[k][3]
        reverse_actions = [a for a in reversed(reverse_actions)]
        start_depth_dict = obs[sym_step_idx]['depth']
        start_mask_dict = obs[sym_step_idx]['mask']
        reverse_action_short = reverse_actions[len(reverse_actions)-sym_step_idx: ]
        sym_depth_image_traj3 = local_sym_step(
            start_depth_dict, start_mask_dict, reverse_action_short, reverse=False, **args
        )
        # obs
        sym_depth_image_traj3 = [s for s in reversed(sym_depth_image_traj3)]
        new_obs = deepcopy(obs)
        for _idx, o in enumerate(sym_depth_image_traj3):
            new_obs[_idx]['image'][0,:,:] = o
        new_sym_obss.append(new_obs)
        # actions
        new_sym_actions = deepcopy(origin_actions)
        action_short = [-a for a in reversed(reverse_action_short)]
        for _idx, a in enumerate(action_short):
            new_sym_actions[_idx] = a
        new_sym_actionss.append(new_sym_actions)

    return new_sym_obss, new_sym_actionss