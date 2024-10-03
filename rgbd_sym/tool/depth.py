# https://github.com/vitalemonate/depth2Cloud/blob/main/depth2Cloud.py
import os
import numpy as np
import cv2
from tqdm import tqdm
from rgbd_sym.tool.common import scale_arr
import time

def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()


def depth_image_to_point_cloud(rgb, depth, scale, K, pose, encode_mask=None, tolist=True):

    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)
    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)
    valid = Z > 0
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    stacklist = (X, Y, Z, np.ones(len(X)))
    position = np.vstack(stacklist)
    position = np.dot(pose, position)
    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]
    if encode_mask is None:
        points = np.transpose(np.vstack((position[0:3, :], R, G, B)))
    else:
        M = np.ravel(encode_mask)[valid]
        points = np.transpose(np.vstack((position[0:3, :], R, G, B, M)))
    if tolist:
        points = points.tolist()
    return points


# image_files: XXXXXX.png (RGB, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# poses: camera-to-world, 4×4 matrix in homogeneous coordinates
def build_point_cloud(dataset_path, scale, view_ply_in_world_coordinate):
    K = np.fromfile(os.path.join(dataset_path, "K.txt"), dtype=float, sep="\n ")
    K = np.reshape(K, newshape=(3, 3))
    image_files = sorted(Path(os.path.join(dataset_path, "images")).files('*.png'))
    depth_files = sorted(Path(os.path.join(dataset_path, "depth_maps")).files('*.png'))

    if view_ply_in_world_coordinate:
        poses = np.fromfile(os.path.join(dataset_path, "poses.txt"), dtype=float, sep="\n ")
        poses = np.reshape(poses, newshape=(-1, 4, 4))
    else:
        poses = np.eye(4)

    for i in tqdm(range(0, len(image_files))):
        image_file = image_files[i]
        depth_file = depth_files[i]

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)

        if view_ply_in_world_coordinate:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[i])
        else:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses)
        save_ply_name = os.path.basename(os.path.splitext(image_files[i])[0]) + ".ply"
        save_ply_path = os.path.join(dataset_path, "point_clouds")

        if not os.path.exists(save_ply_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(save_ply_path)
        write_point_cloud(os.path.join(save_ply_path, save_ply_name), current_points_3D)

def get_intrinsic_matrix(width, height, fov, degrees=True):
    """ Calculate the camera intrinsic matrix.
    """
    if degrees:
        fov = np.deg2rad(fov)
    fy = fx = (width / 2.) / np.tan(fov / 2.)  # fy = fy?
    cx, cy = width / 2., height / 2.
    mat = np.array([[fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]]).astype(np.float)
    return mat

def pointclouds2occupancy(pc_mat, occup_h, occup_w, occup_d, 
                          pc_x_min=None,pc_x_max=None,
                          pc_y_min=None,pc_y_max=None,
                          pc_z_min=None,pc_z_max=None,
                          ):
    _pc_mat = pc_mat.copy()
    # print(_pc_mat[:,0],  pc_x_min, pc_x_max, 0, occup_h-1,)
    get_idx = lambda mat, min, max, occ_s: np.clip(scale_arr(mat, min, max, 0, occ_s-1), 0, occ_s-1).astype(int)
    # idx_x = scale_arr(_pc_mat[:,0],  pc_x_min, pc_x_max, 0, occup_h-1,)
    # idx_y = scale_arr(_pc_mat[:,1],  pc_y_min, pc_y_max, 0, occup_w-1,)
    # idx_z = scale_arr(_pc_mat[:,2],  pc_z_min, pc_z_max, 0, occup_d-1,)
    print(_pc_mat.shape)
    if pc_x_min is None: pc_x_min=np.min(_pc_mat[:,0])
    if pc_x_max is None: pc_x_max=np.max(_pc_mat[:,0])
    if pc_y_min is None: pc_y_min=np.min(_pc_mat[:,1])
    if pc_y_max is None: pc_y_max=np.max(_pc_mat[:,1])
    if pc_z_min is None: pc_z_min=np.min(_pc_mat[:,2])
    if pc_z_max is None: pc_z_max=np.max(_pc_mat[:,2])
    idx_x = get_idx(_pc_mat[:,0],  pc_x_min, pc_x_max,occup_h)
    idx_y = get_idx(_pc_mat[:,1],  pc_y_min, pc_y_max,occup_w)
    idx_z = get_idx(_pc_mat[:,2],  pc_z_min, pc_z_max,occup_d)
    occ_mat = np.zeros((occup_h, occup_w, occup_d), dtype=bool)
    occ_mat[idx_x,idx_y,idx_z] = True
    return occ_mat

def occup2image(occ_mat, image_type='projection',
            pc_x_min=None,pc_x_max=None,
            pc_y_min=None,pc_y_max=None,
            pc_z_min=None,pc_z_max=None,
                ):
    if image_type=="projection":
        occ_proj_y = np.sum(occ_mat, axis=0) !=0
        occ_proj_z = np.flip(np.transpose(np.sum(occ_mat, axis=1) != 0), 0)
        occ_proj_x = np.transpose(np.sum(occ_mat, axis=2) != 0)
        return occ_proj_x, occ_proj_y, occ_proj_z
    elif image_type=="depth":
        s = occ_mat.shape
        index_arr = np.tile(np.arange(s[0]).reshape(-1,1,1,), (1,s[1],s[2]))
        index_arr[np.logical_not(occ_mat)] = s[0]
        x = np.min(index_arr, axis=0)
        x = np.clip(scale_arr(x, 0, s[0]-1, 0, 255),0,255).astype(np.uint8)
        index_arr = np.tile(np.arange(s[1]).reshape(1,-1,1,), (s[0],1,s[2]))
        index_arr[np.logical_not(occ_mat)] = s[1]
        y = np.min(index_arr, axis=1)
        y = np.clip(scale_arr(y, 0, s[1]-1, 0, 255),0,255).astype(np.uint8)
        index_arr = np.tile(np.arange(s[2]).reshape(1,1,-1,), (s[0],s[1],1))
        index_arr[np.logical_not(occ_mat)] = s[2]
        z = np.min(index_arr, axis=2)
        z = np.clip(scale_arr(z, 0, s[2]-1, 0, 255),0,255).astype(np.uint8)
        z = np.transpose(z)
        from matplotlib.pyplot import imshow, subplot, axis, cm, show
        # print(x.shape)
        
        # subplot(1, 3, 1)
        # imshow(x)
        # subplot(1, 3, 2)
        # imshow(y)
        # subplot(1, 3, 3)
        # imshow(z)        
        # show()
        return x, y, z
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # dataset_folder = Path("dataset")
    # scene = Path("hololens")
    # # 如果view_ply_in_world_coordinate为True,那么点云的坐标就是在world坐标系下的坐标，否则就是在当前帧下的坐标
    # view_ply_in_world_coordinate = False
    # # 深度图对应的尺度因子，即深度图中存储的值与真实深度（单位为m）的比例, depth_map_value / real depth = scale_factor
    # # 不同数据集对应的尺度因子不同，比如TUM的scale_factor为5000， hololens的数据的scale_factor为1000, Apollo Scape数据的scale_factor为200
    # scale_factor = 1000.0
    # build_point_cloud(os.path.join(dataset_folder, scene), scale_factor, view_ply_in_world_coordinate)
    from gym_ras.api import make_env
    from pathlib import Path
    env, env_config = make_env(tags=['gasv2_surrol'], seed=0)
    imgs = env.render()
    print(imgs.keys())
    print(imgs['mask'].keys())
    rgb = imgs["rgb"]
    depth = imgs["depReal"]
    encode_mask = np.zeros(depth.shape, dtype=np.uint8)
    masks = [imgs["mask"]["psm1"], imgs["mask"]["stuff"]]
    for i in range(len(masks)):
        encode_mask[masks[i]] = i
    scale = 1
    K = get_intrinsic_matrix(depth.shape[0],depth.shape[1],fov=45) 
    pose = np.eye(4)
    points = depth_image_to_point_cloud(
        rgb, depth, scale, K, pose, encode_mask=encode_mask
    )

    save_ply_name = "test_pc.ply"
    save_ply_path = Path('./data/test_pc') 
    save_ply_path.mkdir(parents=True, exist_ok=True)
    write_point_cloud(str(save_ply_path / save_ply_name), points)
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow, subplot, axis, cm, show
    from gym_ras.tool.common import getT, invT, TxT
    points = np.array(points)
    points = points[points[:,6]==1] # mask out
    T1 = getT([0,0,-0.2*5], [0,0,0], rot_type='euler')
    T2 = getT([0,0,0], [-45,0,0], rot_type='euler', euler_Degrees=True)
    ones = np.ones((points.shape[0],1))
    P = np.concatenate((points[:,:3], ones), axis=1)
    points[:,:3] = np.matmul(P, np.transpose(TxT([T2, T1,])))[:,:3]
    print(points[:,:3])
    occ_proj_x, occ_proj_y, occ_proj_z = pointclouds2occupancy(
        points,
        occup_h=200,
        occup_w=200,
        occup_d=200,
        pc_x_min=-0.1 * 5,
        pc_x_max=0.1 * 5,
        pc_y_min=-0.1 * 5,
        pc_y_max=0.1 * 5,
        pc_z_min=-0.1 * 5,
        pc_z_max=0.1 * 5,
    )
    im = [occ_proj_x, occ_proj_y, occ_proj_z]
    for k in range(len(im)):
        subplot(1, len(im), k + 1)
        axis("off")
        plt.imshow(im[k], cmap="gray")
    plt.show()  
