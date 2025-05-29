from rgbd_sym.env.embodied.pomdp.env import PomdpEnv
from rgbd_sym.env.wrapper.occup import Occup
import gym
import matplotlib.pyplot as plt
from rgbd_sym.tool.plt import plot_img

env = PomdpEnv(task = 'block_pull')
env = Occup(env)
obss = []
masks = []
occups = []
obs = env.reset()
obss.append(obs)


done = False
while not done:
    action = env.get_oracle_action()
    action[4] = 0
    obs, reward, done, info = env.step(action)
    obss.append(obs)

plt_data = [[o['image'][0,:,:] for o in obss]]

for k, v in obss[0]['mask'].items():
    _mask = [o['mask'][k] for o in obss]
    plt_data.append(_mask)

plt_data.append([o['occup_image'] for o in obss])

plot_img(plt_data)


