from rgbd_sym.api import make_env
from rgbd_sym.tool.common import getT, TxT, scale_arr
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
from rgbd_sym.tool.depth import get_intrinsic_matrix, depth_image_to_point_cloud, pointclouds2occupancy, occup2image
from rgbd_sym.tool.sym import local_depth_transform
import numpy as np
import cv2
env1, env_config = make_env(tags=[], seed=0)
env2, env_config = make_env(tags=[], seed=0)
obs1 = env1.reset()
# obs1 = env1.reset()
obs2 = env2.reset()

obss = [obs1, obs2]
for i, obs in enumerate(obss):
    _m = None
    for k,v in obs['mask'].items():
        _m = np.logical_or(_m, v) if _m is not None else v
    obs['depthReal'][np.logical_not(_m)] = 1
    depth = scale_arr(obs['depthReal'], 0,1, 0 ,255)
    ax = subplot(1, len(obss), i+1)
    imshow(depth)
    plt.colorbar()
    ax.set_title(f"obs {i+1}")

show()