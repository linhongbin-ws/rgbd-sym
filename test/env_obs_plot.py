from rgbd_sym.api import make_env
from rgbd_sym.env.wrapper import Visualizer, ActionOracle
import argparse
from tqdm import tqdm
import numpy as np
import time
parser = argparse.ArgumentParser()
parser.add_argument('--env-tag', type=str, nargs='+', default=['block_pick'])
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


env, env_config = make_env(tags=args.env_tag, seed=0)
# env = ActionOracle(env, device="script")
for i in range(args.seed):
    obs = env.reset()

done = False
obs = env.reset()
obs = env.reset()

image_list = []
mask_list = []
# original_depth_list = []
cnt = 0
image_list.append(obs['image'])
while not done:
    action = env.get_oracle_action()
    obs, reward, done, info = env.step(action)
    img = obs['image']
    # img =  np.transpose(img, axes=[1, 2, 0])
    # img = np.uint8(img* 255)
    # img = np.concatenate((img, np.zeros(img.shape[:2]+(1,),dtype=np.uint8)), axis=2, dtype=np.uint8)
    image_list.append(img)

    # original_depth_list.append(obs['orgin_depth_image'])
    cnt+=1
    print(cnt)

from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['figure.figsize'] = [50, 40]
image_list = image_list[:]
for i in range(len(image_list)):
    ax = subplot(1, len(image_list), 1+i)
    imshow(image_list[i][0,:,:])
    plt.clim(0,0.3)
    plt.colorbar()

for i in range(len(image_list)):
    ax = subplot(2, len(image_list), 1+i+len(image_list))
    imshow(image_list[i][1,:,:,])
    
    plt.colorbar()
# original_depth_list = original_depth_list[:]
# for i in range(len(original_depth_list)):
#     ax = subplot(2, len(original_depth_list), 1+i+len(original_depth_list))
#     imshow(original_depth_list[i][:,:])
#     plt.colorbar()
show()