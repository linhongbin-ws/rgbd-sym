from gym.envs.registration import register

register(
    "BlockPulling-Symm-v0",
    entry_point="rgbd_sym.env.embodied.pomdp.block_pulling:BlockEnv",
    max_episode_steps=50,
)
