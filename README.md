
# Download 

```sh
git clone https://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@github.com/linhongbin-ws/rgbd-sym.git -b devel
cd rgbd-sym
git submodule update --init --recursive
```

# Install 

## Conda install (DreamerV2, RSAC)

- Install [miniconda](https://docs.anaconda.com/miniconda/)

- edit init bash [init.sh](./bash/init.sh)
  
- Create conda virtual environment
    ```sh
    source bash/init.sh
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.9 -y
    ```
- source init
    ```sh
    source ./bash/init.sh 
    ```
- Install torch
    ```sh
    conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 pytorch
    ```
- Install the dependency `equi-rl-for-pomdps` for RSAC and POMDP environment
    ```sh
    pushd ext/equi-rl-for-pomdps && python -m pip install -r requirements.txt && popd
    pushd ext/equi-rl-for-pomdps/escnn/ && python -m pip install -r requirements.txt && python -m pip install -e . && popd
    pushd ext/equi-rl-for-pomdps/pomdp_robot_domains/ && python -m pip install -r requirements.txt && python -m pip install -e . && popd
    pushd ext/equi-rl-for-pomdps/pomdp-domains/ &&  python -m pip install -e . && popd
    ```
- Install dreamerv2
    ```sh
    pushd ext/dreamerv2/ && python -m pip install -e . && popd # install dreamerv2
    ```
- Install `rgbd-sym`
    ```
    python -m pip install -e . 
    ```

## Conda install (DreamerV3)

- Install [miniconda](https://docs.anaconda.com/miniconda/)

- edit init bash [bash/init_dv3.sh](./bash/init_dv3.sh)
  
- Create conda virtual environment
    ```sh
    source bash/init_dv3.sh
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.10 -y
    ```
- source init
    ```sh
    source bash/init_dv3.sh 
    ```
<!-- - Install torch
    ```sh
    conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 pytorch
    ``` -->
- Install the dependency `equi-rl-for-pomdps` for POMDP environment
    ```sh
    pushd ext/equi-rl-for-pomdps && python -m pip install -r requirements.txt && popd
    pushd ext/equi-rl-for-pomdps/escnn/ && python -m pip install -r requirements.txt && python -m pip install -e . && popd
    pushd ext/equi-rl-for-pomdps/pomdp_robot_domains/ && python -m pip install -r requirements.txt && python -m pip install -e . && popd
    pushd ext/equi-rl-for-pomdps/pomdp-domains/ &&  python -m pip install -e . && popd
    ```
- Install Dreamerv3
    ```sh
    pushd ./ext/dreamerv3/ &&  python -m pip install -U -r embodied/requirements.txt && popd
    pushd ./ext/dreamerv3/ &&  python -m pip install -U -r dreamerv3/requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && popd
    ```

- Install `rgbd-sym`
    ```
    python -m pip install -e . 
    ```

# Run

## Train RSAC

```sh
source bash/init.sh
python ./ext/equi-rl-for-pomdps/policies/main.py --cfg ./ext/equi-rl-for-pomdps/configs/block_picking/rnn.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 80
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pulling/rnn.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 80
```

## Train Dreamerv2
```sh
source bash/init.sh
python ./run/rl.py --baseline-tag pomdp  --baseline dreamerv2
```
import pybullet as pb
import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_pomdp_block_pulling_planner import CloseLoopPomdpBlockPullingPlanner
from helping_hands_rl_envs.pybullet.utils.constants import NoValidPositionException

class CloseLoopPomdpBlockPullingEnv(CloseLoopEnv):
  def __init__(self, config):
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.8, 0.8]
    super().__init__(config)

  def reset(self, target_obj_idx, noise=False):
    while True:
      self.resetPybulletWorkspace()
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
      try:
        if not self.random_orientation:
          padding = self._getDefaultBoarderPadding(constants.FLAT_BLOCK)
          min_distance = self._getDefaultMinDistance(constants.FLAT_BLOCK)
          x = np.random.random() * (self.workspace_size - padding) + self.workspace[0][0] + padding/2
          while True:
            y1 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding/2
            y2 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding/2
            if max(y1, y2) - min(y1, y2) > min_distance:
              break
          self._generateShapes(constants.FLAT_BLOCK, 1, pos=[[x, y1, 0.025]], random_orientation=True, cube_color='blue')
          self._generateShapes(constants.FLAT_BLOCK, 1, pos=[[x, y2, 0.025]], random_orientation=True)
        else:
          self._generateShapes(constants.FLAT_BLOCK, 2, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    return self.objects[0].isTouching(self.objects[1])

  def getEnvPenalty(self):
    xy_diff = np.array(self.objects[0].getXYPosition()) - np.array(self.objects[1].getXYPosition())
    xy_distance = np.linalg.norm(xy_diff)
    return -xy_distance + 0.1

def createCloseLoopPomdpBlockPullingEnv(config):
  return CloseLoopPomdpBlockPullingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0.01, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False}
  env_config['seed'] = 1
  env = CloseLoopPomdpBlockPullingEnv(env_config)
  planner = CloseLoopPomdpBlockPullingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  # while True:
  #   current_pos = env.robot._getEndEffectorPosition()
  #   current_rot = transformations.euler_from_quaternion(env.robot._getEndEffectorRotation())
  #
  #   block_pos = env.objects[0].getPosition()
  #   block_rot = transformations.euler_from_quaternion(env.objects[0].getRotation())
  #
  #   pos_diff = block_pos - current_pos
  #   rot_diff = np.array(block_rot) - current_rot
  #   pos_diff[pos_diff // 0.01 > 1] = 0.01
  #   pos_diff[pos_diff // -0.01 > 1] = -0.01
  #
  #   rot_diff[rot_diff // (np.pi/32) > 1] = np.pi/32
  #   rot_diff[rot_diff // (-np.pi/32) > 1] = -np.pi/32
  #
  #   action = [1, pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]]
  #   obs, reward, done = env.step(action)

  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)

  # fig, axs = plt.subplots(8, 5, figsize=(25, 40))
  # for i in range(40):
  #   action = planner.getNextAction()
  #   obs, reward, done = env.step(action)
  #   axs[i//5, i%5].imshow(obs[2][0], vmax=0.3)
  # env.reset()
  # fig.show()
## Train Dreamerv3
```sh
source ./bash/init_dv3.sh
python ./run/train_dreamerv3.py
```
>> bug fixing: if you encounter issue: `ImportError: cannot import name 'Mapping' from 'collections'`, 
>> you need to modify scripts in `miniconda3/envs/rgbd-sym-dv3/lib/python3.10/collections`from
>> `from collections import Mapping, MutableMapping, Sequence`
>> to
>>  `from collections.abc import Mapping, MutableMapping, Sequence`
>> See bugs in [stackoverflow](https://stackoverflow.com/questions/69381312/importerror-cannot-import-name-from-collections-using-python-3-10)

## Env play
- Init conda environment
    ```sh
    source bash/init.sh # for RSAC dreamerv2
    source ./bash/init_dv3.sh # for dreamerv3
    ```
- Play environment with demonstration script

    ```sh
    python ./run/env_play.py 
    ```
    press any key to proceed steps, press `q` to quit

## Simulated trajectory test

- Init conda environment
    ```sh
    source bash/init.sh # for RSAC dreamerv2
    source ./bash/init_dv3.sh # for dreamerv3
    ```
- simulated trajectory with local symmetric transform
    ```sh
    python ./test/local_sym8.py
    ```

<!-- ## Train baselines
- Init conda environment
    ```sh
    source ./bash/init.sh # for drearmerv2
    ```
- Train dreamerv2
  ```
  python ./run/rl.py --baseline-tag pomdp --baseline dreamerv2 --env-tag pomdp
  ``` -->