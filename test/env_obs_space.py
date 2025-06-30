from rgbd_sym.api import make_env
env = make_env('block_push')
print(env.observation_space)