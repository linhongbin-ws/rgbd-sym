
import open3d as o3d
import numpy as np

def depth_image_to_point_cloud( rgb, depth_real, scale, new_K, pose, encode_mask=None, tolist=False):
    width = depth_real.shape[0]
    height = depth_real.shape[1]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=width, height=height, fx=new_K[0][0], fy=new_K[1][1], cx=new_K[0][2], cy=new_K[1][2])
    # intrinsic = o3d.core.Tensor(K[:3][:3])
    # print(np.max(depth),np.min(depth))

    scale_depth = 255
    depth_image = depth_real*scale_depth
    depth_image = depth_image.astype(np.uint16)
    color_raw = rgb.copy()
    if encode_mask is not None:
        color_raw[:,:,2] = encode_mask.astype(np.uint8)
        # print("color_raw[:,:,2]",color_raw[:,:,2])

    color_raw = np.asarray(color_raw, order="C")

    color_raw = o3d.geometry.Image(color_raw)
    depth_raw = o3d.geometry.Image(depth_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=scale_depth, depth_trunc=10000000,
                                                               convert_rgb_to_intensity=False)

    pointSet = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, 
                                            intrinsic,project_valid_depth_only=False)
    

    points = np.asarray(pointSet.points)
    colors = np.asarray(pointSet.colors)
    mat = np.concatenate((points,colors), axis=1)
    if encode_mask is not None:
        mat = np.concatenate((mat, mat[:,5:6]*255), axis=1)
    if tolist:
        mat = mat.tolist()
    return mat