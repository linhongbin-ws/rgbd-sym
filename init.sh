source ~/ssd/anaconda3/bin/activate
conda activate rgbd-sym
export PYTHONPATH=${PWD}:$PYTHONPATH
export LD_LIBRARY_PATH=/home/ben/ssd/anaconda3/envs/rgbd-sym/lib/:/usr/local/lib/
sudo nvidia-smi -pl 300
