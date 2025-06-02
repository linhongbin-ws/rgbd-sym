from rgbd_sym.env.wrapper import Occup, Sym
from rgbd_sym.env.embodied.pomdp.env import PomdpEnv
from rgbd_sym.env.embodied.dummy.env import DummyEnv
from rgbd_sym.env.wrapper.occup import Occup
from rgbd_sym.tool.plt import plot_img, plot_traj
from rgbd_sym.tool.common import getT, TxT
from rgbd_sym.tool.sym import generate_sym3, get_sym_params, action_inverse
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='block_pick')
args = parser.parse_args()

env = PomdpEnv(task = args.task)
env = Occup(env)

dummy_env = DummyEnv()
dummy_env = Occup(dummy_env)


env = Sym(env, dummy_env)
env.set_sym(True)

new_sym_obss = []
new_sym_actionss = []
for i in range(5):
    obs = env.reset()
    done = False
    obss_origin = []
    obss_origin.append(obs)
    actions = []
    sym_action = []
    while not done:
        action = env.get_oracle_action()
        obs,reward, done, info = env.step(action)
        obss_origin.append(obs)
        actions.append(action)
        sym_action.append(obs['sym_action'])
    new_sym_obss.append(obss_origin)
    if obs['sym_state']:
        new_sym_actionss.append(sym_action)
    else:
        actions_origin = actions


imgss = []
for _new_obs in new_sym_obss:
    imgss.append([{"image":  v['occup_image'], "title": f"Sym step {i}"} for i, v in enumerate(_new_obs)])
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [50, 40]
plot_img(imgss)



def a2T(a):

    T = getT(a[1:4].tolist(),[0,0,0,], rot_type="euler")
    return T

def aggTs(Ts):
    T = getT([0,0,0],[0,0,0,], rot_type="euler")
    current_T = T.copy()
    newTs = [current_T]
    for _T in Ts:
        current_T = TxT([_T, current_T])
        newTs.append(current_T)
    return newTs

points_mats = []
trajTs = [a2T(action_inverse(a)) for a in reversed(actions_origin)]
trajTs = aggTs(trajTs)
points_mats = []
pc_mat = {}
print(trajTs[0])
pc_mat["mat"] = np.array([[t[0][3], t[1][3], t[2][3]] for t in trajTs])
pc_mat["linewidth"] = 4
pc_mat["alpha"] = 1
pc_mat["label"] = f"original traj"
pc_mat["color"] = "k"
points_mats.append(pc_mat)

for i, new_actions in enumerate(new_sym_actionss):
    new_trajTs = [a2T(action_inverse(a)) for a in reversed(new_actions)]
    new_trajTs = aggTs(new_trajTs)
    pc_mat = {}
    pc_mat["mat"] = np.array([[t[0][3], t[1][3], t[2][3]] for t in new_trajTs])
    pc_mat["linewidth"] = 4
    pc_mat["alpha"] = 0.5
    pc_mat["label"] = f"traj {i+1}"
    points_mats.append(pc_mat)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [50, 40]   
plot_traj(points_mats, elev=45, azim=135, roll=0)
