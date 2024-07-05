from rgbd_sym.api import make_env
from rgbd_sym.env.wrapper import Visualizer
import numpy as np
env, config = make_env(tags=[])
env = Visualizer(env,keyboard=True)

obs = env.reset()
# env.cv_show(obs[:,:,])
done = False

while not done:
    env.cv_show(obs)
    action = env.get_oracle_action(obs)
    obs, reward, done, info = env.step(action)
    print(obs)

print(env.observation_space)
print(env.action_space)
