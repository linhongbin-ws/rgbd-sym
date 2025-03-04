source bash/init.sh 
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 0 --sym_normal 0
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 60 --sym_normal 0 