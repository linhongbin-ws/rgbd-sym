from rgbd_sym.api import make_env
from rgbd_sym.tool.common import getT, TxT
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
from rgbd_sym.tool.depth import get_intrinsic_matrix, depth_image_to_point_cloud, pointclouds2occupancy, occup2image
import numpy as np
import cv2
env, env_config = make_env(tags=[], seed=0)
obs = env.reset()


def _resize_bool(im, size):
    _in = np.zeros(im.shape, dtype=np.uint8)
    _in[im] = 1
    _out = cv2.resize(_in, (size, size))
    return _out == 1
fov = 45
done = False
gripper_project_offset = 0.2 # gripper is z zero, so projection is not in FOV45, we need to somehow recover
obss = []
while not done:
    action = env.get_oracle_action()
    obs, reward, done, info = env.step(action)
    K = get_intrinsic_matrix(obs['depthReal'].shape[0], obs['depthReal'].shape[1], fov=45)
    pose = np.eye(4)
    rgb = obs['rgb']
    depth = obs['depthReal']
    depth[obs['mask']['gripper']] +=gripper_project_offset # gripper 
    scale = 1
    print(depth.shape)
    encode_mask = np.zeros(depth.shape, dtype=np.uint8)
    masks = [obs['mask'][k] for k in obs['mask']]
    for i, v in enumerate(masks):
        subplot(1, len(masks), i + 1)
        axis("off")
        imshow(v)
    show()
    print(depth[masks[0]])
    mask_key = [k for k in obs['mask']]
    # print(mask_key)
    for m_id, m in enumerate(masks):
        encode_mask[m] = m_id + 1
    points = depth_image_to_point_cloud(
        rgb, depth, scale, K, pose, encode_mask=encode_mask, tolist=False
    )
    points[points[:,6] == 1,2] -=  gripper_project_offset
    print(np.unique(points[:, 6]))
    # T1 = getT([0, 0, -0.2 * 5], [0, 0, 0], rot_type="euler")
    # T2 = getT([0, 0, 0], [-45, 0, 0], rot_type="euler", euler_Degrees=True)
    # ones = np.ones((points.shape[0], 1))
    # P = np.concatenate((points[:, :3], ones), axis=1)
    # points[:, :3] = np.matmul(P,np.transpose(TxT([T2,T1,])))[:, :3]
    obss.append(obs)
    occup_imgs = {}
    ws_scale = 0.08
    x_offset = +0.00
    y_offset = -0.00
    z_offset = 0.06
    z_scale=8
    pc_x_min=-ws_scale+x_offset
    pc_x_max=ws_scale+x_offset
    pc_y_min=-ws_scale+y_offset
    pc_y_max=ws_scale+y_offset
    pc_z_min=-ws_scale+z_offset
    pc_z_max=ws_scale*z_scale+z_offset
    for m_id, _ in enumerate(masks):
        print(np.unique(points[:, 6]))
        _points = points[points[:, 6] == m_id + 1]  # mask out
        if len(_points) == 0:
            print("skip mask",m_id + 1)
            continue
        occ_mat = pointclouds2occupancy(
            _points,
            occup_h=40,
            occup_w=40,
            occup_d=40,
            pc_x_min=pc_x_min,
            pc_x_max=pc_x_max,
            pc_y_min=pc_y_min,
            pc_y_max=pc_y_max,
            pc_z_min=pc_z_min,
            pc_z_max=pc_z_max,
        )
        x, y, z = occup2image(occ_mat, image_type='depth',
                                pc_x_min=pc_x_min,
            pc_x_max=pc_x_max,
            pc_y_min=pc_y_min,
            pc_y_max=pc_y_max,
            pc_z_min=pc_z_min,
            pc_z_max=pc_z_max,)
        del occ_mat
        # x = _resize_bool(x, obs["rgb"].shape[0])
        # y = _resize_bool(y, obs["rgb"].shape[0])
        # z = _resize_bool(z, obs["rgb"].shape[0])
        print(x.shape)
        x = cv2.resize(x, (obs["rgb"].shape[0], obs["rgb"].shape[0]))
        y = cv2.resize(y, (obs["rgb"].shape[0], obs["rgb"].shape[0]))
        z = cv2.resize(z, (obs["rgb"].shape[0], obs["rgb"].shape[0]))
        occup_imgs[mask_key[m_id]] = [x, y, z]
    break
length = len(obss)

imgs = [v for k, v in occup_imgs.items()]
print(occup_imgs)
for i, v in enumerate(imgs):
    subplot(3, len(imgs), i + 1)
    axis("off")
    imshow(v[0])
    subplot(3, len(imgs), i + 1 + len(imgs)*1)
    axis("off")
    imshow(v[1])
    subplot(3, len(imgs), i + 1 + len(imgs)*2)
    axis("off")
    imshow(v[2])
show()


# encode_mask = np.zeros(imgs[0][0].shape, dtype=np.uint8)
# for m_id, m in enumerate(imgs):
#     # print(encode_mask.shape,m[0])
#     encode_mask[m[0]] = m_id + 1
# for m_id, m in enumerate(masks):
#     encode_mask[m] = m_id + 1 + len(imgs)
# imshow(encode_mask)
# show()
z_image = np.zeros(masks[0].shape,dtype=np.uint8)
for m_id, v in enumerate(imgs):
    z_image += v[2]
# for m_id, m in enumerate(masks):
#     z_image[m] = 255
axis("on")
subplot(1, 3, 1)
imshow(z_image,vmin=0, vmax=255)
plt.colorbar()
subplot(1, 3, 2)
imshow(obs['image'][:,:,0],vmin=0, vmax=255)
plt.colorbar()
z_image2 = np.zeros(masks[0].shape,dtype=np.uint8)
z_image2 +=z_image
z_image2+=obs['image'][:,:,0]
subplot(1, 3, 3)
imshow(z_image2,vmin=0, vmax=255)
plt.colorbar()
show()

