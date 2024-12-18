from rgbd_sym.api import make_env
from rgbd_sym.tool.plt import plot_img, get_backend, use_backend


moveable = True
imgss = []

env, env_config = make_env(tags=["block_pull"], seed=0)
obs = env.reset()
origin_actions = []
imgs1_meta = [obs]
done = False
while not done:
    if moveable:
        action = env.query_expert(1)
    else:
        action = env.query_expert(2)
    origin_actions.append(action)
    obs, reward, done, info = env.step(action)
    imgs1_meta.append(obs)
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][0,:,:] for o in imgs1_meta])])
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][1,:,:] for o in imgs1_meta])])

env, env_config = make_env(tags=["block_pull", "no_sym_obs"], seed=0)
obs = env.reset()
origin_actions = []
imgs1_meta = [obs]
done = False
while not done:
    if moveable:
        action = env.query_expert(1)
    else:
        action = env.query_expert(2)
    origin_actions.append(action)
    obs, reward, done, info = env.step(action)
    imgs1_meta.append(obs)
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][0,:,:] for o in imgs1_meta])])
imgss.append([{"image": v, "title": f"step {i}"} for i, v in enumerate([o['image'][1,:,:] for o in imgs1_meta])])


use_backend('tkagg')
plot_img(imgss, )
print("backend:", get_backend())
