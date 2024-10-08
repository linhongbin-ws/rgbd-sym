from pdomains import *
import matplotlib.pyplot as plt
import gym
from pomdp_envs import pomdp 

env = gym.make("BlockPicking-Symm-v0", rendering=True)
obs = env.reset()

cnt = 0
ep_idx = 0

for i in range(1000):
    # action = env.action_space.sample()
    action = env.query_expert(ep_idx)
    obs, reward, done, info = env.step(action)
    print(obs.shape)
    cnt += 1
    if done:
        print(cnt)
        print(info)
        ep_idx += 1
        cnt = 0
        env.reset()
