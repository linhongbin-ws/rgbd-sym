from pdomains import *
from rgbd_sym.env.embodied.base import BaseEnv
import gym

class PomdpEnv(BaseEnv):
    def __init__(self,
                 task,
                pybullet_gui=False,
                  **kwargs,):
        if task== 'block_picking':
            task_id = 'pdomains-block-picking-v0'
        client=gym.make(task_id, rendering=pybullet_gui)
        super().__init__(client)

    def reset(self):
        self.timestep = 0
        obs = self.client.reset()
        return obs
    
    def step(self, action):
        self.timestep += 1
        obs, reward, done, info = self.client.step(action)
        return obs, reward, done, info

    def render(self, mode="human"):  # ['human', 'rgb_array', 'mask_array']
        return self.client.render(mode=mode)

    def get_oracle_action(self, obs):
        return self.client.query_expert(0)

if __name__ == "__main__":
    env = POMDP_ENV(task_name='pdomains-block-picking-v0')
    obs = env.reset()
    done = False

    while not done:
        # action = env.action_space.sample()
        action = env.get_oracle_action(obs)
        obs, reward, done, info = env.step(action)
        print(obs)
    print(env.observation_space)
    print(env.action_space)