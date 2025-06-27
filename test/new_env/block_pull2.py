from rgbd_sym.env.embodied.pomdp.env import PomdpEnv
from rgbd_sym.env.wrapper.occup import Occup
import gym
import matplotlib.pyplot as plt
from rgbd_sym.tool.plt import plot_img
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='block_pull')
args = parser.parse_args()

env = PomdpEnv(task = args.task)
env = Occup(env)
obss = []
obs = env.reset()
obss.append(obs)

import sys
sys.path.append("./ext/equi-rl-for-pomdps/pomdp-domains/")
if args.task == "block_pull":
    from pdomains.block_pulling import BlockEnv
    clss = BlockEnv
    query_id = 0
elif args.task == "block_pick":
    from pdomains.block_picking import BlockEnv
    clss = BlockEnv
    query_id = 0

elif args.task == "block_push":
    from pdomains.block_pushing import BlockEnv
    clss = BlockEnv
    query_id = 1

elif args.task == "drawer_open":
    from pdomains.drawer_opening import DrawerEnv
    clss = DrawerEnv
    query_id = 0


done = False
while not done:
    action = env.query_expert(query_id)
    obs, reward, done, info = env.step(action)
    obss.append(obs)

plt_data = [[o['image'][0,:,:] - np.min(o['image'][0,:,:])  for o in obss]]

for k, v in obss[0]['mask'].items():
    _mask = [o['mask'][k] for o in obss]
    plt_data.append(_mask)

plt_data.append([o['occup_image'] for o in obss])




env = clss()
obss = []
obs = env.reset()
obs = env.reset()
obss.append(obs)


done = False
while not done:
    action = env.query_expert(query_id)
    obs, reward, done, info = env.step(action)
    # print(obs)
    obss.append(obs)

plt_data.append([o[0,:,:] for o in obss])

plot_img(plt_data)


