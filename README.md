
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
```

## Train Dreamerv2
```sh
source bash/init.sh
python ./run/rl.py --baseline-tag pomdp  --baseline dreamerv2
```

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