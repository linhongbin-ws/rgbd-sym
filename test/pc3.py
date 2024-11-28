from rgbd_sym.tool.o3d import depth_image_to_point_cloud
# from rgbd_sym.tool.depth import depth_image_to_point_cloud
import open3d as o3d
from rgbd_sym.api import make_env
import numpy as np

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
    pointSet = o3d.geometry.PointCloud()
    pointSet.points = o3d.utility.Vector3dVector(pcd[:,:3].tolist())
    color_mat  = np.ones(pcd[:,:3].shape)
    for i in range(color_mat.shape[1]):
         color_mat[:,i]  = color_mat[:,i]*np.random.uniform(0,1)
    pointSet.colors = o3d.utility.Vector3dVector(color_mat)
    # pointSet.colors = o3d.utility.Vector3dVector(pcd[:,3:6].tolist())
    print("type", type(pointSet))
    print(np.max(depth),np.min(depth),)
    print(np.max(np.asarray(pointSet.points)),np.min(np.asarray(pointSet.points)))
    pc.append(pointSet)
    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pointSet,
                                                                voxel_size=0.01)
    print(voxel_grid.get_voxels())
    vx.append(voxel_grid)
o3d.visualization.draw_geometries(pc)
o3d.visualization.draw_geometries(vx)

