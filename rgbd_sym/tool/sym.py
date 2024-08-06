import numpy as np
import cv2
from rgbd_sym.tool.common import scale_arr, getT
import numpy as np
from scipy.ndimage import affine_transform


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
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    if dxy is not None:
        rotated_dxy = rot.dot(dxy)
        rotated_dxy = np.clip(rotated_dxy, -1, 1)
    else:
        rotated_dxy = None
    # Apply rigid transform to image and pixel labels.

    if not action_only:
        current_image = affine_transform(
            current_image, np.linalg.inv(transform), mode="nearest", order=1
        )
        if next_image is not None:
            next_image = affine_transform(
                next_image, np.linalg.inv(transform), mode="nearest", order=1
            )
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
