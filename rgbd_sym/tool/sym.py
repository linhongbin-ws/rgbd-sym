import numpy as np
import cv2
from rgbd_sym.tool.common import scale_arr, getT, TxT
import numpy as np
from scipy.ndimage import affine_transform

# local sym dependency
from rgbd_sym.tool.depth import get_intrinsic_matrix, depth_image_to_point_cloud, pointclouds2occupancy, occup2image

def get_random_transform_params(image_size, trans_scale=1, rot_scale=1):
    theta = np.random.random() * 2 * np.pi
    trans = np.random.randint(0, image_size[0] // 10, 2) - image_size[0] // 20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    trans = trans_scale * trans
    theta = rot_scale * theta
    return theta, trans, pivot


def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array(
        [[1.0, 0.0, -pivot[0]], [0.0, 1.0, -pivot[1]], [0.0, 0.0, 1.0]]
    )
    image_t_pivot = np.array(
        [[1.0, 0.0, pivot[0]], [0.0, 1.0, pivot[1]], [0.0, 0.0, 1.0]]
    )
    transform = np.array(
        [
            [np.cos(theta), -np.sin(theta), trans[0]],
            [np.sin(theta), np.cos(theta), trans[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))


def perturb(
    current_image,
    next_image,
    action,
    theta,
    trans,
    pivot,
    set_theta_zero=False,
    set_trans_zero=False,
    action_only=False,
):
    """Perturn an image for data augmentation"""

    # image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    # theta, trans, pivot = get_random_transform_params(image_size)

    dxy = action[1:3]

    if set_theta_zero:
        theta = 0.0
    if set_trans_zero:
        trans = [0.0, 0.0]
    # transform = get_image_transform(theta, trans, pivot)

    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    if dxy is not None:
        rotated_dxy = rot.T.dot(dxy) # transpose, fix bug of action roation
        rotated_dxy = np.clip(rotated_dxy, -1, 1)
    else:
        rotated_dxy = None
    # Apply rigid transform to image and pixel labels.
    p = np.array(pivot)
    # print(pivot)
    offset = p - p.dot(rot)
    s = current_image.shape
    if not action_only:
        image_list = []
        for i in range(s[2]):
            image_list.append(affine_transform(
                current_image[:,:,i],
                rot.T,
                mode="nearest",
                offset=offset,
                order=1,
                output_shape=(s[0], s[1]),
                output=np.uint8,
            ))
        current_image = np.stack(image_list, axis=2)
        if next_image is not None:
            image_list = []
            for i in range(s[2]):
                image_list.append(
                    affine_transform(
                        next_image[:, :, i],
                        rot.T,
                        mode="nearest",
                        offset=offset,
                        order=1,
                        output_shape=(s[0], s[1]),
                        output=np.uint8,
                    )
                )
            next_image = np.stack(image_list, axis=2)
    new_action = action.copy()
    new_action[1:3] = rotated_dxy
    return current_image, next_image, new_action, transform_params


def RGBDTransform(rgb, depth_real, fx, fy, cx,cy, Ts=[]):
    assert len(Ts)!=0
    height, width , channel = rgb.shape[0], rgb.shape[1],rgb.shape[2]
    zs = depth_real.copy().reshape(-1)
    rs = rgb[:,:,0].copy().reshape(-1)
    gs = rgb[:,:,1].copy().reshape(-1)
    bs = rgb[:,:,2].copy().reshape(-1)
    rgb_u = np.stack(width*[np.arange(width)], axis=1)
    rgb_v = np.stack(height*[np.arange(height)], axis=0)
    rgb_u_arr = rgb_u.reshape(-1)
    rgb_v_arr = rgb_v.reshape(-1)
    xs = np.multiply((rgb_u_arr - cy) / (fx),  zs)
    ys = np.multiply((rgb_v_arr - cx) / (fy),  zs)
    
    new_rgbs = []
    for T in Ts:
        assert T.shape[0]==4 and  T.shape[1]==4
        xyz = np.stack([xs,ys,zs, np.ones(xs.shape)], axis=0)
        print(T.shape, xyz.shape)
        new_xyz = np.matmul(T , xyz)
        new_x = new_xyz[0,:]
        new_y = new_xyz[1,:]
        new_z = new_xyz[2,:]
        print(fx * np.divide(new_x, new_z) + cy)
        new_us = fx * np.divide(new_x, new_z) + cy
        new_vs = fy * np.divide(new_y, new_z) + cx
        new_us = np.around(new_us).astype(np.int)
        new_vs = np.around(new_vs).astype(np.int)

        new_us = new_us.reshape(-1)
        new_vs = new_vs.reshape(-1)
        new_rgb = np.zeros(( width, height, channel,),dtype=np.uint8)
        new_depth = -np.ones(( width, height,),dtype=np.float)
        for i in range(new_us.shape[0]):
            u = new_us[i]
            v = new_vs[i]
            if (u<0) or (u>width-1): continue
            if (v<0) or (v>height-1): continue
            if new_depth[u,v] >= zs[i]: continue

            new_rgb[u,v,0] = rs[i]
            new_rgb[u,v,1] = gs[i]
            new_rgb[u,v,2] = bs[i]
        new_rgbs.append(new_rgb)
    return new_rgbs


if __name__ == "__main__":
    rgb = cv2.imread('./asset/20240326-124513-rgb.png')
    depth = cv2.imread('./asset/20240326-124513-depth.png')

    width = 600
    height = 600
    fov = 42/180*np.pi
    fx = width / 2 / np.tan(fov/2)
    fy = fx
    cx = (width-1) / 2
    cy = (height-1) / 2
    

    depth_real = scale_arr(depth[:,:,0], 0,255,1.25, 1.75)

    T0 = getT([0,0,0,], [0,0,10], rot_type="euler", euler_convension="xyz", euler_Degrees=True)
    Ts = [T0]
    new_rgbs = RGBDTransform(rgb,depth_real,fx,fy, cx,cy, Ts)


    for new_rgb in new_rgbs:
        cv2.imshow('image',new_rgb)
        cv2.waitKey(0)


def local_depth_transform(depth, mask_dict, K, depth_real_min, depth_real_max,
                           gripper_project_offset,
                           transform_dict,
                            pc_x_min,
                            pc_x_max,
                            pc_y_min,
                            pc_y_max,
                            pc_z_min,
                            pc_z_max,
                            occup_h,
                            occup_w,
                            occup_d,
                            background_encoding=255
                           ):
    depth_real = scale_arr(depth, 0, 255, depth_real_min, depth_real_max) # depth image to depth
    depth_real[mask_dict['gripper']] +=gripper_project_offset # gripper is depth zero, so we need to offset it in a camera view, otherwise the gripper shape is weird 
    # print(depth_real[mask_dict['gripper']])
    # print(depth_real[mask_dict['object1']])
    encode_mask = np.zeros(depth.shape, dtype=np.uint8)
    # masks = [mask_dict[k] for k in mask_dict]
    # for i, v in enumerate(masks):
    #     subplot(1, len(masks), i + 1)
    #     axis("off")
    #     imshow(v)
    # show()
    # mask_key = [k for k in mask_dict]
    masks = []
    encode_id = {}
    m_id = 0
    for k,v in mask_dict.items():
        m_id+=1
        masks.append(k)
        encode = m_id + 1
        encode_mask[v] =encode
        encode_id[k] = encode
    # for m_id, m in enumerate(masks):
    #     encode_mask[m] = m_id + 1 # background 0, other mask key 1, 2, 3 ...
    scale = 1
    pose = np.eye(4)
    rgb = np.zeros(depth.shape + (3,),dtype=np.uint8)
    points = depth_image_to_point_cloud(
        rgb, depth_real, scale, K, pose, encode_mask=encode_mask, tolist=False
    )
    points[points[:,6] == encode_id['gripper'] ,2] -=  gripper_project_offset # recover gripper depth from offset to zero.
    print(np.unique(points[:, 6]))

    for k,v in transform_dict.items():
        pc_idx = points[:,6] == encode_id[k] 
        ones = np.ones((points[pc_idx,:].shape[0], 1))
        P = np.concatenate((points[pc_idx, :3], ones), axis=1)
        points[pc_idx, :3] = np.matmul(P,np.transpose(v))[:, :3]
                
    # T1 = getT([0, 0, -0.2 * 5], [0, 0, 0], rot_type="euler")
    # T2 = getT([0, 0, 0], [-45, 0, 0], rot_type="euler", euler_Degrees=True)
    # ones = np.ones((points.shape[0], 1))
    # P = np.concatenate((points[:, :3], ones), axis=1)
    # points[:, :3] = np.matmul(P,np.transpose(TxT([T2,T1,])))[:, :3]
    # images = []
    # new_depth = np.zeros(depth.shape, dtype=np.uint8)
    # for m_id, _ in enumerate(masks):
    #     # print(np.unique(points[:, 6]))
    #     _points = points[points[:, 6] == m_id + 1]  # mask out
    #     if len(_points) == 0:
    #         print("skip mask",m_id + 1)
    #         continue
    points = points[points[:,6] != 0, :] # remove background
    occ_mat = pointclouds2occupancy(
        points,
        occup_h=occup_h,
        occup_w=occup_w,
        occup_d=occup_d,
        pc_x_min=pc_x_min,
        pc_x_max=pc_x_max,
        pc_y_min=pc_y_min,
        pc_y_max=pc_y_max,
        pc_z_min=pc_z_min,
        pc_z_max=pc_z_max,
    )
    z = occup2image(occ_mat, image_type='depth',background_encoding=background_encoding) 
    del occ_mat
    s = depth.shape
    z = cv2.resize(z, (s[0], s[1]),interpolation=cv2.INTER_NEAREST)
    return z

     