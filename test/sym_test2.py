from rgbd_sym.api import make_env
from rgbd_sym.env.wrapper import Visualizer, ActionOracle, SymAug
from rgbd_sym.tool.img_tool import save_img
import argparse
from tqdm import tqdm
import time
parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')

parser.add_argument('-p', type=int)
parser.add_argument('--repeat', type=int, default=5)
parser.add_argument('--action', type=str, default="oracle")
# parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
# parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument('--oracle', type=str, default='script')
parser.add_argument('--no-vis', action="store_true")
parser.add_argument('--sym', action="store_true")
parser.add_argument('--eval', action="store_true")
parser.add_argument('--savedir', type=str, default="./data/sym_test")

args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)


eps_idx =0
for _ in tqdm(range(args.repeat)):
    done = False
    obs = env.reset()
    step=0
    eps_idx+=1
    if env.IsSymEnv:
        save_img(obs["image"], args.savedir + '/' + str(eps_idx), str(step))
    while not done:
        step+=1
        if env.IsSymEnv:
            action = env.get_sym_action()
            obs, reward, done, info = env.step(action)
            print("input action", action, "output", info["action"])
            save_img(obs["image"], args.savedir + '/' + str(eps_idx), str(step))
        else:
            action = env.get_oracle_action()
            obs, reward, done, info = env.step(action)
