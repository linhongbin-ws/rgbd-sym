from rgbd_sym.tool.sym import generate_sym3, get_sym_params, action_inverse
# from gym_ras.tool.plt import use_backend, plot_img, plot_traj

from rgbd_sym.tool.common import getT, TxT
from rgbd_sym.env.embodied.pomdp.env import PomdpEnv
from rgbd_sym.env.embodied.dummy.env import DummyEnv
from rgbd_sym.env.wrapper.occup import Occup
import gym
import matplotlib.pyplot as plt
from rgbd_sym.tool.plt import plot_img, plot_traj
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='block_pick')
args = parser.parse_args()

env = PomdpEnv(task = args.task)
env = Occup(env)
obss = []
obs = env.reset()

obss_origin = []
obss_origin.append(obs)
actions_origin = []
done = False

gripper_state = 0
while not done:
    gripper_state-= 0.2
    action = env.get_oracle_action()
    action[0] = gripper_state
    obs,reward, done, info = env.step(action)
    obss_origin.append(obs)
    actions_origin.append(action)
    # print("gripper_pos z: ",obs['gripper_pos'][2])


dummy_env = DummyEnv(                 
                delta_transl = 1,
                 delta_rot =  np.pi / 8)
dummy_env = Occup(dummy_env)

sym_trans_z = 1
sym_trans_r = 1
sym_trans_rots = np.linspace(0, np.pi * 2, 4, endpoint=False).tolist()
sym_rot = 0
new_sym_obss = []
new_sym_actionss = []
for sym_trans_rot in sym_trans_rots:
    new_sym_obs, new_sym_actions = generate_sym3(obss_origin,actions_origin,
                                                sym_end_step=None, 
                                                sym_trans_z=sym_trans_z, 
                                                sym_trans_r=sym_trans_r, 
                                                sym_trans_rot=sym_trans_rot, 
                                                sym_rot=sym_rot,
                                                dummy_env=dummy_env)
    new_sym_obss.append(new_sym_obs)
    new_sym_actionss.append(new_sym_actions)




# for i in range(len(actions_origin)):
#     # print(f"actions_origin err: {actions_origin[i]-new_sym_actions[i]}")
#     print(actions_origin[i][1:4])




imgss = []
imgss.append([{"image": v['occup_image'], "title": f"GT step {i}"} for i, v in enumerate(obss_origin)])
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
# print(trajTs[0])
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