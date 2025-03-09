# Simulation

## 1 Dem + block pull
- NIEA + equi-RSAC

- equi-RSAC + SNIA
```sh
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 6 --sym_normal 0 
```

- equi-RSAC
```sh
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pull/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 0 --sym_normal 0 
```


## 1 Dem + block pick
- NIEA + equi-RSAC

- equi-RSAC + SNIA
```sh
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pick/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 6 --sym_normal 0 
```

- equi-RSAC
```sh
python ./rgbd_sym/rl/rsac/main.py --cfg ./rgbd_sym/rl/rsac/configs/block_pick/rnn-equi-all.yml --algo sac --seed 0 --cuda 0 --num_expert_episodes 1 --sym_expert 0 --sym_normal 0 
```

