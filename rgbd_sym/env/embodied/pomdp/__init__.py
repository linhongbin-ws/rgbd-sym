from gym.envs.registration import register

register(
    "BlockPull-Sym",
    entry_point="rgbd_sym.env.embodied.pomdp.block_pulling:BlockEnv",
    max_episode_steps=50,
)

register(
    "BlockPick-Sym",
    entry_point="rgbd_sym.env.embodied.pomdp.block_picking:BlockEnv",
    max_episode_steps=50,
)

register(
    "BlockPush-Sym",
    entry_point="rgbd_sym.env.embodied.pomdp.block_pushing:BlockEnv",
    max_episode_steps=50,
)

register(
    "DrawerOpen-Sym",
    entry_point="rgbd_sym.env.embodied.pomdp.drawer_opening:DrawerEnv",
    max_episode_steps=50,
)


