source bash/init.sh
gpu_id=${1:-0}
seed=${2:-0}
demon=${3:-80}
CUDA_VISIBLE_DEVICES=${gpu_id} python ./rgbd_sym/rl/rsac/main.py \
 --cfg ./rgbd_sym/rl/rsac/configs/drawer_open/rnn-equi-all.yml \
 --algo sac --seed ${seed} --cuda 0 --num_expert_episodes ${demon} \
 --sym_expert 4 --sym_normal 0 --traj_batch 3 --prefix bmt