from rgbd_sym.tool.o3d import depth_image_to_point_cloud, pointclouds2occupancy
# from rgbd_sym.tool.depth import depth_image_to_point_cloud
from rgbd_sym.tool.depth import occup2image
import open3d as o3d
from rgbd_sym.api import make_env
import numpy as np
from rgbd_sym.tool.plt import plot_img


gripper_project_offset = 0.2 # gripper is z zero, so projection is not in FOV45, we need to somehow recover
ws_scale = 0.2
x_offset = 0.0
y_offset = 0.0
z_offset = 0.8
pc_x_min=-ws_scale+x_offset
pc_x_max=ws_scale+x_offset
pc_y_min=-ws_scale+y_offset
pc_y_max=ws_scale+y_offset
pc_z_min=-ws_scale+z_offset
pc_z_max=ws_scale+z_offset
occup_h=84 
occup_w=84 
occup_d=84

env, env_config = make_env(tags=['block_pull', "no_clutch"], seed=0)
obs = env.reset()
action = np.array([0.0,0,0, -1,0])

steps = 6
obss = []
for i in range(steps):
    obs, reward, done, info = env.step(action)
    obss.append(obs)





pc = []
vx = []
for obs in obss:
    rgb = obs['rgb']
    depth = obs['depth']['object1']

    encode_key = ['object1']
    encode_mask = np.zeros(depth.shape, dtype=np.uint8)
    encode_id = {}
    m_id = 0
    for k,v in obs['mask'].items():
        if k in encode_key:
            m_id+=1
            encode = m_id
            encode_mask[v] =encode
            encode_id[k] = encode


    scale = 1
    proj_matrix = env.get_projection_matrix()
    proj_matrix = np.array(list(proj_matrix)).reshape(4,4)
    K = np.zeros((3,3))
    size = 84
    K[0][0] = proj_matrix[0,0] * size /2
    K[1][1] = proj_matrix[1,1] * size /2
    K[0][2] = size/2
    K[1][2] = size/2
    print("K is ", K)
    pose = np.eye(4,4)
    pcd = depth_image_to_point_cloud(rgb, depth, scale, K, pose, encode_mask=encode_mask, tolist=False)
    print("mask id", pcd[:,6])
    pcd = pcd[pcd[:,6]==1]
    print("point clouds range", np.min(pcd[:,:3], axis=0),np.max(pcd[:,:3], axis=0),)
    occ_mat = pointclouds2occupancy(pc_mat=pcd,
        occup_h=occup_h,
        occup_w=occup_w,
        occup_d=occup_d,
        pc_x_min=pc_x_min,
        pc_x_max=pc_x_max,
        pc_y_min=pc_y_min,
        pc_y_max=pc_y_max,
        pc_z_min=pc_z_min,
        pc_z_max=pc_z_max,)

    px,py,pz = occup2image(occ_mat, image_type='projection',background_encoding=255) 
    
    plot_img([[px,py,pz,depth]])
