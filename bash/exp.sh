source bash/init.sh 

# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 320 --sym_expert 0 --sym_normal 0 --prefix WSBen-aug0

python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pick/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 \
--num_expert_episodes 80 --sym_expert 4 --traj_batch 3  --sym_normal 0 --prefix laptop

# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --num_init_episodes 1 --sym_expert 0 --sym_normal 0 --prefix laptop

# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_push/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 320 --sym_expert 0 --sym_normal 0 --prefix WSBen-aug0

# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/drawer_open/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 320 --sym_expert 0 --sym_normal 0 --prefix WSBen-aug0


# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 80 --sym_expert 0 --sym_normal 0 --prefix laptop
# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pick/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 6 --sym_normal 0  
# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_push/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 6 --sym_normal 0 
# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_push/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 0 --sym_normal 0  
# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/c/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 6 --sym_normal 0  
# python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/drawer_open/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 0 --sym_normal 0 