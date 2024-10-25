# Install 
## 1.1. Download 

```sh
git clone https://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@github.com/linhongbin-ws/rgbd-sym.git -b devel
cd rgbd-sym
git submodule update --init --recursive
```

## Miniconda Install

- Install [miniconda](https://docs.anaconda.com/miniconda/)

- Edit file for environment variables [config.sh](./config/config.sh)

- source config files
    ```sh
    source ./config/config.sh 
    ```
- Create conda virtual environment
    ```sh
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.9 -y
    ```
- source init files again
    ```sh
    source ./bash/init.sh 
    ```
- Install the dependency `equi-rl-for-pomdps`
    ```sh
    pushd ext/equi-rl-for-pomdps && python -m pip install -r requirements.txt && popd
    pushd ext/equi-rl-for-pomdps/pomdp_robot_domains/ && python -m pip install -r requirements.txt && python -m pip install -e . && popd
    pushd ext/equi-rl-for-pomdps/pomdp-domains/ &&  python -m pip install -e . && popd
    pushd ext/equi-rl-for-pomdps &&  python -m pip install -e . && popd
    ```
- Install `rgbd-sym`
    ```
    python -m pip install -e . 
    ```


# Run

## Env play
- Init conda environment
    ```sh
    source ./bash/init.sh 
    ```
- Play environment with demonstration script

    ```sh
    python ./run/env_play.py 
    ```
    press any key to proceed steps.

- Backtrace with sym

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