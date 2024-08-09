from rgbd_sym.api import make_env
from rgbd_sym.env.wrapper import Visualizer, ActionOracle
import argparse
from tqdm import tqdm
import time
from pathlib import Path
from rgbd_sym.tool.sym import get_random_transform_params, perturb
import cv2

parser = argparse.ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",
)

parser.add_argument("-p", type=int)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--action", type=str, default="oracle")
# parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
# parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument("--env-tag", type=str, nargs="+", default=[])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--vis-tag", type=str, nargs="+", default=[])
parser.add_argument("--oracle", type=str, default="script")
parser.add_argument("--no-vis", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--savedir", type=str, default="./data/sym_test")
parser.add_argument("--sym", type=int, default=4)

args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)


eps = 1
done = False
env.reset()
step = 0
while not done:
    step+=1
    if any(i.isdigit() for i in args.action):
        action = int(args.action)
    elif args.action == "random":
        action = env.action_space.sample()
    elif args.action == "oracle":
        action = env.get_oracle_action()
    obs, reward, done, info = env.step(action)
    path = Path(args.savedir) / str(eps) / "origin"
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path / (str(step) + ".jpg")), cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR)
    )

    for k in range(args.sym):
        path = Path(args.savedir) / str(eps) / ("sym" + str(k+1))
        theta, trans, pivot = get_random_transform_params(obs['image'][0].shape)
        obs_new, _, action_new, _ = perturb(
            obs["image"],
            None,
            action,
            theta,
            trans,
            pivot,
            set_theta_zero=False,
            set_trans_zero=True,
            action_only=False,
        )
        path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(path / (str(step) + ".jpg")), cv2.cvtColor(obs_new, cv2.COLOR_RGB2BGR)
        )
