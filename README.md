# rgbd-sym

## Download 

```sh
git clone https://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@github.com/linhongbin-ws/rgbd-sym.git
cd rgbd-sym
git submodule update --init --recursive
```

## 2.2. Conda Install

- We use virtual environment in [Anancoda](https://www.anaconda.com/download) to install our codes. Make sure you install anaconda.
```sh
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
```

- Edit environment variables, go to [config.sh](./config.sh) and edit your environment variables.

- Create conda virtual environment
    ```sh
    source ./config/config.sh
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.9 -y
    ```

- Install our package 
    ```sh
    source bash/init.sh
    conda install cudnn=8.2 cudatoolkit=11.3 libffi==3.3 ffmpeg -c anaconda -c conda-forge -y
    pushd ext/equi-rl-for-pomdps && python -m pip install -r requirements.txt && popd
    pushd ext/equi-rl-for-pomdps/pomdp_robot_domains/ && python -m pip install -r requirements.txt && python -m pip install -e . && popd
    pushd ext/equi-rl-for-pomdps/pomdp-domains/ &&  python -m pip install -e . && popd
    pushd ext/equi-rl-for-pomdps &&  python -m pip install -e . && popd
    pushd ext/dreamerv2/ && python -m pip install -e . && popd # install dreamerv2
    python -m pip install -e . 
    ```
    > Our GPU Dependency in Anaconda: cudatoolkit=11.3, cudnn=8.2,tensorflow=2.9.0 tensorflow_probability=0.17.0. Other versions should work, while user need to check the compatability of cuda and tensorflow.  
- Install dreamerv3
    ```sh
    source ./config/config_dreamerv3.sh
    source $ANACONDA_PATH/bin/activate 
    source bash/init_dreamerv3.sh
    conda create -n $ENV_NAME python=3.9 -y
    conda install nvidia/label/cuda-12.3.2::cuda -y
    conda install -c anaconda cudnn=9 -y
    pip install -U "jax[cuda12]"
    ```