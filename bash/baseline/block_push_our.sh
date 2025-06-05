source bash/init.sh
prefix=${1:-" "}
gpu_id=${2:-0}
seed=${3:-0}
demon=${4:-80}
mea_expert=${5:-12}
mea_normal=${6:0}
CUDA_VISIBLE_DEVICES=${gpu_id} python ./rgbd_sym/rl/main.py \
 --cfg configs/block_push/mea-rnn-equi-all.yml \
 --algo sac --seed ${seed} --cuda 0 --num_expert_episodes ${demon} \
 --prefix ${prefix} \
 --mea_expert ${mea_expert} --mea_normal ${mea_normal}