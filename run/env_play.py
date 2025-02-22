from rgbd_sym.api import make_env
from rgbd_sym.env.wrapper import Visualizer, ActionOracle
import argparse
from tqdm import tqdm
import numpy as np
import time
parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')

parser.add_argument('-p', type=int)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--action', type=str, default="oracle")
# parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
# parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument('--oracle', type=str, default='script')
parser.add_argument('--no-vis', action="store_true")
parser.add_argument('--eval', action="store_true")

args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)
print(env_config)
if args.action == 'oracle':
    env = ActionOracle(env, device=args.oracle)
if not args.no_vis:
    env = Visualizer(env, update_hz=100 if args.action in [
                     'oracle'] else -1, vis_tag=args.vis_tag, keyboard=not args.action in ['oracle'])

if args.eval:
    env.to_eval()
print("action space:" , env.action_space)
print("observation space:" , env.observation_space)
def get_depth_image(depth_dict):
    depths = [v for k,v in depth_dict.items()]
    new_obs_depth = np.min(np.stack(depths, axis=0), axis=0)
    return new_obs_depth

for _ in tqdm(range(args.repeat)):
    done = False
    obs = env.reset()
    if not args.no_vis:
        img = obs.copy()
        # img['image'] = np.concatenate((img['image'], np.zeros(img['image'].shape[:2]+(1,),dtype=np.uint8)), axis=2, dtype=np.uint8)
        if isinstance(img, dict):
            img.pop("image", None)
            img.pop("depth", None)
            img.pop("depthR", None)
            img.pop("imageR", None)
        else:
            if img.shape[0] <3:
                img = np.transpose(img, axes=[2,1,0])
                img = img * 255
                img = img.astype(np.uint8)
                pad_dim = 3 - img.shape[2]

                img = np.concatenate(
                    (img, np.zeros(img.shape[:2] + (pad_dim,),dtype=np.uint8)), axis=2
                )
                img = {"rgb": img}

        img_break = env.cv_show(imgs=img)
        # img_break = env.cv_show(imgs=img)
    # print("obs:", obs)
    while not done:
        # action = env.action_space.sample()
        # print(action)
        print("==========step", env.timestep, "===================")
        if any(i.isdigit() for i in args.action):
            action = np.array([0.0,0,0,0,0])
            action[int(args.action)] = 1
        elif args.action == "random":
            action = env.action_space.sample()
        elif args.action == "oracle":
            action = env.get_oracle_action()
            print(action)
        else:
            raise NotImplementedError
        # print("step....")
        obs, reward, done, info = env.step(action)

       
        print(obs)
        img = obs.copy()

        if isinstance(img, dict):
            print(img.keys())
            img["depth"] = get_depth_image(img["depth"])
            img.pop("image", None)
            img.pop("imageR", None)
            img.pop("depthR", None)
        else:
            if img.shape[0] < 3:
                img = np.transpose(img, axes=[2, 1, 0])
                # img = img[:,:,0:]
                img = img * 255
                img = img.astype(np.uint8)
                pad_dim = 3 - img.shape[2]

                img = np.concatenate(
                    (img, np.zeros(img.shape[:2] + (pad_dim,), dtype=np.uint8)), axis=2
                )
                img = {"rgb": img}

        # img['image'] = np.concatenate((img['image'], np.zeros(img['image'].shape[:2]+(1,),dtype=np.uint8)), axis=2, dtype=np.uint8)
        # img.pop("depth", None)
        print(img.keys())
        print(info)
        print("is exceed ws:",obs["is_exceed_ws"])
        print("reward:", reward, "done:", done,)


        if not args.no_vis:
            img_break = env.cv_show(imgs=img)
            if img_break:
                break
    if not args.no_vis:
        if img_break:
            break
