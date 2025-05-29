from envs import pomdp
import gym
import matplotlib.pyplot as plt
from rgbd_sym.tool.plt import plot_img
env = gym.make("BlockPulling-Symm-v0")
obss = []
obs = env.reset()
obss.append(obs[0,:,:])
done = False
while not done:
    action = env.query_expert(0)
    obs, reward, done, info = env.step(action)
    obss.append(obs[0,:,:])

plot_img([obss])


