source bash/init.sh 
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pulling/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 80 --sym_expert 6 --sym_normal 0 --prefix equinet
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pulling/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 80 --sym_expert 12 --sym_normal 0 --prefix equinet
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pulling/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 80 --sym_expert 24 --sym_normal 0 --prefix equinet
