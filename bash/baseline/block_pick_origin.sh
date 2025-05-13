source bash/init.sh
prefix=${1:-" "}
rotaug=${2:-4}
gpu_id=${3:-0}
seed=${4:-0}
demon=${5:-80}
CUDA_VISIBLE_DEVICES=${gpu_id} python ./rgbd_sym/rl/rsac/main.py \
 --cfg ./rgbd_sym/rl/rsac/configs/block_pick/rnn-equi-all.yml \
 --algo sac --seed ${seed} --cuda 0 --num_expert_episodes ${demon} \
 --sym_expert 0 --sym_normal 0 --traj_batch 0 --prefix ${prefix} \
 --rotaug ${rotaug}