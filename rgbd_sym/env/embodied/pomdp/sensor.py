import pybullet as pb
import numpy as np

class Sensor(object):
  def __init__(self, cam_pos, cam_up_vector, target_pos, target_size, near, far):
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=cam_pos,
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )

    self.near = near
    self.far = far
    self.fov = np.degrees(2 * np.arctan((target_size / 2) / self.far))
    self.proj_matrix = pb.computeProjectionMatrixFOV(self.fov, 1, self.near, self.far)

  def setCamMatrix(self, cam_pos, cam_up_vector, target_pos):
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=[cam_pos[0], cam_pos[1], cam_pos[2]],
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    self.proj_matrix = pb.computeProjectionMatrixFOV(70, 1, 0.001, 0.3)

  def getHeightmap(self, size):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    depth_img = np.array(image_arr[3])
    depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)

    return np.abs(depth - np.max(depth)).reshape(size, size)

  def getPointCloud(self, size, to_numpy=True):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix,
                                  renderer=pb.ER_TINY_RENDERER)
    depthImg = np.asarray(image_arr[3])

    # https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
    projectionMatrix = np.asarray(self.proj_matrix).reshape([4,4],order='F')
    viewMatrix = np.asarray(self.view_matrix).reshape([4,4],order='F')
    tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
    pixel_pos = np.mgrid[0:size, 0:size]
    pixel_pos = pixel_pos/(size/2) - 1
    pixel_pos = np.moveaxis(pixel_pos, 1, 2)
    pixel_pos[1] = -pixel_pos[1]
    zs = 2*depthImg.reshape(1, size, size) - 1
    pixel_pos = np.concatenate((pixel_pos, zs))
    pixel_pos = pixel_pos.reshape(3, -1)
    augment = np.ones((1, pixel_pos.shape[1]))
    pixel_pos = np.concatenate((pixel_pos, augment), axis=0)
    position = np.matmul(tran_pix_world, pixel_pos)
    pc = position / position[3]
    points = pc.T[:, :3]

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    # if to_numpy:
      # points = np.asnumpy(points)
    return points
  
  def getHeightmap2(self, size):
    image_arr = pb.getCameraImage(width=size, height=size,
                                    viewMatrix=self.view_matrix,
                                    projectionMatrix=self.proj_matrix,
                                    renderer=pb.ER_BULLET_HARDWARE_OPENGL,
                                    flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
    depth_img = np.array(image_arr[3])
    depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)
    # depth = np.abs(depth - np.max(depth)).reshape(size, size)
  


    mask = image_arr[4].reshape(size, size)
    mask_background_id=-5
    in_obj_data = np.bitwise_and(mask, ((1 << 24) - 1))
    in_link_data = (np.right_shift(mask, 24)) - 1
    in_obj_data[mask < 0] = mask_background_id
    in_link_data[mask < 0] = mask_background_id
    mask_metadata = (in_obj_data, in_link_data)

    rgb = image_arr[2]

    img = {}
    img['depth'] = depth
    img['mask_metadata'] = mask_metadata
    img['rgb'] = rgb[:,:,:3]
    img['view_mat'] = self.view_matrix
    img['proj_mat'] = self.proj_matrix
    return img
  

    image_arr = pb.getCameraImage(width=size, height=size,
                                viewMatrix=self.view_matrix,
                                projectionMatrix=self.proj_matrix,
                                renderer=pb.ER_TINY_RENDERER)
    depth_img = np.array(image_arr[3])
    depth = self.far * self.near / (self.far - (self.far - self.near) * depth_img)

    return np.abs(depth - np.max(depth)).reshape(size, size)