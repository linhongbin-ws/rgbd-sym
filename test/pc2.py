from rgbd_sym.tool.depth import depth_image_to_point_cloud
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
from rgbd_sym.tool.plt import plot_img
plot_img([[obs['depth']['object1'] for obs in obss]])
pc = []
for obs in obss:
    rgb = obs['rgb']
    depth = obs['depth']['object1']


    scale = 1
    proj_matrix = env.get_projection_matrix()
    proj_matrix = np.array(list(proj_matrix)).reshape(4,4)
    K = np.zeros((3,3))
    size = 84
    K[0][0] = proj_matrix[0,0] * size /2
    K[1][1] = proj_matrix[1,1] * size /2
    # K[0][2] = size*(1-proj_matrix[0][2])/2
    # K[1][2] = size*(1+proj_matrix[1][2])/2
    print("K is ", K)
    pose = np.eye(4,4)
    pcd = depth_image_to_point_cloud(rgb, depth, scale, K, pose, encode_mask=None, tolist=False)
    pointSet = o3d.geometry.PointCloud()
    pointSet.points = o3d.utility.Vector3dVector(pcd[:,:3].tolist())
    # pointSet.colors = o3d.utility.Vector3dVector(pcd[:,3:6].tolist())
    print("type", type(pointSet))
    print(np.max(depth),np.min(depth),)
    print(np.max(np.asarray(pointSet.points)),np.min(np.asarray(pointSet.points)))
    pc.append(pointSet)
o3d.visualization.draw_geometries(pc)

pc = []
for obs in obss:
    depth = obs['depth']['object1']

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=84, height=84, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])
    # intrinsic = o3d.core.Tensor(K[:3][:3])
    # print(np.max(depth),np.min(depth))
    depth_image = depth*255
    # print(np.max(depth_image),np.min(depth_image))
    depth_image = depth_image.astype(np.uint16)
    # print(np.max(depth_image),np.min(depth_image))
    depth_image = o3d.geometry.Image(depth_image)
    pointSet = o3d.geometry.PointCloud.create_from_depth_image(depth_image, 
                                            intrinsic,depth_scale=255,depth_trunc=1000000,project_valid_depth_only=False)
    pc.append(pointSet)
    print("type", type(pointSet))
    print(np.max(np.asarray(pointSet.points)),np.min(np.asarray(pointSet.points)))
o3d.visualization.draw_geometries(pc)