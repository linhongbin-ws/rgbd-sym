source bash/init.sh
prefix=${1:-" "}
gpu_id=${2:-0}
seed=${3:-0}
demon=${4:-80}
CUDA_VISIBLE_DEVICES=${gpu_id} python ./rgbd_sym/rl/rsac/main.py \
 --cfg ./rgbd_sym/rl/rsac/configs/block_push/rnn-equi-all.yml \
 --algo sac --seed ${seed} --cuda 0 --num_expert_episodes ${demon} \
 --sym_expert 0 --sym_normal 0 --traj_batch 0 --prefix ${prefix}