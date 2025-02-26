## Demonstration-0

- NIEA-Equi-RSAC
```sh
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 0 --sym_expert 0 --sym_normal 6 
```

- Equi-RSAC
```sh
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 0 --sym_expert 0 --sym_normal 0 
```

- RSAC
```sh
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 0 --sym_expert 0 --sym_normal 0 
```

- DrQ-Shift-RSAC
```sh
python policies/main.py --cfg configs/block_pulling/rnn.yml --algo sac_drq --num_rotations 4 --num_expert_episodes 0 --sym_expert 0 --sym_normal 0 
```