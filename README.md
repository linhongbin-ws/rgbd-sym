# rgbd-sym

## Download 

```sh
git clone https://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@github.com/linhongbin-ws/rgbd-sym.git
cd rgbd-sym
git submodule update --init --recursive
```

## 2.2. Conda Install

- We use virtual environment in [Anancoda](https://www.anaconda.com/download) to install our codes. Make sure you install anaconda.

- Edit environment variables, go to [config.sh](./config.sh) and edit your environment variables.

- Create conda virtual environment
    ```sh
    source ./config.sh
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.9 -y
    ```

- Install our gym package 
    ```sh
    source init.sh
    conda install cudnn=8.2 cudatoolkit=11.3 libffi==3.3 ffmpeg -c anaconda -c conda-forge -y
    pushd ext/SurRoL/ && python -m pip install -e . && popd # install surrol
    pushd ext/dreamerv2/ && python -m pip install -e . && popd # install dreamerv2
    python -m pip install -e . # install gym_ras
    ```
    > Our GPU Dependency in Anaconda: cudatoolkit=11.3, cudnn=8.2,tensorflow=2.9.0 tensorflow_probability=0.17.0. Other versions should work, while user need to check the compatability of cuda and tensorflow.  