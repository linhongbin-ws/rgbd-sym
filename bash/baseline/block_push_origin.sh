source bash/init.sh
gpu_id=${1:-0}
seed=${2:-0}
demon=${3:-80}
CUDA_VISIBLE_DEVICES=${gpu_id} python ./rgbd_sym/rl/rsac/main.py \
 --cfg ./rgbd_sym/rl/rsac/configs/block_push/rnn-equi-all.yml \
 --algo sac --seed ${seed} --cuda 0 --num_expert_episodes ${demon} \
 --sym_expert 0 --sym_normal 0 --traj_batch 0 --prefix bmt