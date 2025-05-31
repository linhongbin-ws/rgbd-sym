from rgbd_sym.env.wrapper import Occup, Sym
from rgbd_sym.env.embodied.pomdp.env import PomdpEnv
from rgbd_sym.env.embodied.dummy.env import DummyEnv
from rgbd_sym.env.wrapper.occup import Occup
from rgbd_sym.tool.plt import plot_img, plot_traj


env = PomdpEnv(task = 'block_pull')
env = Occup(env)

dummy_env = DummyEnv()
dummy_env = Occup(dummy_env)


env = Sym(env, dummy_env)
env.set_sym(True)

new_sym_obss = []
for i in range(5):
    obs = env.reset()
    done = False
    obss_origin = []
    obss_origin.append(obs)
    actions_origin = []
    while not done:
        action = env.get_oracle_action()
        obs,reward, done, info = env.step(action)
        obss_origin.append(obs)
        actions_origin.append(action)
    new_sym_obss.append(obss_origin)


imgss = []
for _new_obs in new_sym_obss:
    imgss.append([{"image":  v['occup_image'], "title": f"Sym step {i}"} for i, v in enumerate(_new_obs)])
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [50, 40]
plot_img(imgss)

